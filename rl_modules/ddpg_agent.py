# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on the HeR implementation at https://github.com/TianhongDai/hindsight-experience-replay.

import pickle
import torch
import os
import time
from datetime import datetime
import numpy as np
import hydra
import wandb
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic, actor_tfm, critic_tfm
from rl_modules.normalizer import normalizer, ArrayNormalizer
from her_modules.her import her_sampler


def linear_sched(x0, x1, y0, y1, x):
    m = (y1 - y0) / (x1 - x0)
    return m * (x - x1) + y1


"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, eval_env, env_params, compute_reward, ckpt_data=None):
        self.args = args
        self.env = env
        self.eval_env = eval_env
        self.env_params = env_params

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, compute_reward)
        if ckpt_data is None:
            self.current_epoch = 0
            self.tot_samples = 0
            self.best_success_rate = 0

            # create the network
            self.actor_network = hydra.utils.instantiate(args.actor, env_params)
            self.critic_network = hydra.utils.instantiate(args.critic, env_params)
            # build up the target network
            self.actor_target_network = hydra.utils.instantiate(args.actor, env_params)
            self.critic_target_network = hydra.utils.instantiate(args.critic, env_params)
            # load the weights into the target networks
            self.actor_target_network.load_state_dict(self.actor_network.state_dict())
            self.critic_target_network.load_state_dict(self.critic_network.state_dict())
            # if use gpu
            if self.args.cuda:
                self.actor_network.cuda()
                self.critic_network.cuda()
                self.actor_target_network.cuda().eval()
                self.critic_target_network.cuda().eval()
            # create the optimizer
            self.actor_optim = hydra.utils.instantiate(args.optim_actor, self.actor_network.parameters())
            self.critic_optim = hydra.utils.instantiate(args.optim_critic, self.critic_network.parameters())
            self.actor_sched, self.critic_sched = None, None
            if args.warmup_actor > 0:
                self.actor_sched = torch.optim.lr_scheduler.LambdaLR(self.actor_optim, lambda t: min((t+1) / args.warmup_actor, 1))
            if args.warmup_critic > 0:
                self.critic_sched = torch.optim.lr_scheduler.LambdaLR(self.critic_optim, lambda t: min((t+1) / args.warmup_critic, 1))
            # create the replay buffer
            self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
            # create the normalizer
            self.x_norm = ArrayNormalizer(self.env_params, default_clip_range=self.args.clip_range)
            # create the dict for store the model

            self.init_data = None
            if args.init_trajs:
                print(f"loading initial trajectories from {args.init_trajs}.")
                data = np.load(args.init_trajs)
                self.init_data = [data["grip"], data["obj"], data["ag"], data["g"], data["action"]]
        else:
            self.actor_network = ckpt_data["actor_network"]
            self.critic_network = ckpt_data["critic_network"]
            self.actor_target_network = ckpt_data["actor_target_network"]
            self.critic_target_network = ckpt_data["critic_target_network"]
            self.current_epoch = ckpt_data["current_epoch"]
            self.tot_samples = ckpt_data["tot_samples"]
            self.best_success_rate = ckpt_data["best_success_rate"]
            self.actor_optim = ckpt_data["actor_optim"]
            self.critic_optim = ckpt_data["critic_optim"]
            self.actor_sched = ckpt_data["actor_sched"]
            self.critic_sched = ckpt_data["critic_sched"]
            self.buffer = ckpt_data["buffer"]
            self.x_norm = ckpt_data["x_norm"]

        self.model_dir = os.path.join(wandb.run.dir, "models")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def learn(self):
        """
        train the network

        """
        if self.current_epoch == 0 and self.init_data is not None:
            self.buffer.store_episode(self.init_data)
            self._update_normalizer(self.init_data)
            for _ in range(self.args.n_init_steps):
                for _ in range(self.args.n_batches):
                    metrics = self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
        while self.current_epoch < self.args.n_epochs:
            if self.args.exp_schedule:
                start_slope, end_slope, final_ratio = self.args.exp_schedule
                assert final_ratio < 1 and start_slope < end_slope
                exp_ratio = np.clip(linear_sched(start_slope, end_slope, 1, final_ratio, self.current_epoch), final_ratio, 1)
                noise_eps = self.args.noise_eps * exp_ratio
                random_eps = self.args.random_eps * exp_ratio
            else:
                noise_eps, random_eps = self.args.noise_eps, self.args.random_eps
            start = time.time()
            for _ in range(self.args.n_cycles):
                mb_grip, mb_obj, mb_ag, mb_g, mb_actions = [], [], [], [], []
                for _ in range(self.args.num_rollouts):
                    # reset the rollouts
                    ep_grip, ep_obj, ep_ag, ep_g, ep_actions = [], [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        grip, obj = observation['gripper_arr'], observation['object_arr']
                        g = observation['desired_goal_arr']
                        with torch.no_grad():
                            inputs = self._preproc_inputs(grip, obj, g)
                            pi = self.actor_network(*inputs)
                            action = self._select_actions(pi, noise_eps, random_eps)
                        # feed the actions into the environment
                        observation_new, _, _, _ = self.env.step(action)
                        # append rollouts
                        ep_grip.append(grip)
                        ep_obj.append(obj.copy())
                        ep_ag.append(observation['achieved_goal_arr'].copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        observation = observation_new
                    ep_grip.append(observation['gripper_arr'].copy())
                    ep_obj.append(observation['object_arr'].copy())
                    ep_ag.append(observation['achieved_goal_arr'].copy())

                    mb_grip.append(np.stack(ep_grip, 1))
                    mb_obj.append(np.stack(ep_obj, 1))
                    mb_ag.append(np.stack(ep_ag, 1))
                    mb_g.append(np.stack(ep_g, 1))
                    mb_actions.append(np.stack(ep_actions, 1))
                # convert them into arrays
                mb_grip = np.concatenate(mb_grip, 0)
                mb_obj = np.concatenate(mb_obj, 0)
                mb_ag = np.concatenate(mb_ag, 0)
                mb_g = np.concatenate(mb_g, 0)
                mb_actions = np.concatenate(mb_actions, 0)
                self.tot_samples += mb_actions.shape[0] * mb_actions.shape[1]
                # store the episodes
                self.buffer.store_episode([mb_grip, mb_obj, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_grip, mb_obj, mb_ag, mb_g, mb_actions])
                if self.tot_samples > self.args.min_samples:
                    for _ in range(self.args.n_batches):
                        # train the network
                        metrics = self._update_network()
                    # soft update
                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)
                    wandb.log(metrics, step=self.tot_samples)
            # start to do the evaluation
            success_rate, ret, vid_path = self._eval_agent()
            print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), self.current_epoch, success_rate))
            print(f'Epoch time: {time.time() - start}')
            wandb.log({
                'success_rate': success_rate,
                'return': ret,
                'epoch': self.current_epoch,
                'vids/0': wandb.Video(vid_path, format="mp4"),
                'exploration/random_eps': random_eps,
                'exploration/noise_eps': noise_eps,
            }, step=self.tot_samples)
            save_data = [self.x_norm, self.actor_network]
            torch.save(save_data, os.path.join(self.model_dir, "latest.pt"))
            if success_rate >= self.best_success_rate or self.current_epoch == 0:
                self.best_success_rate = success_rate
                torch.save(save_data, os.path.join(self.model_dir, "best.pt"))
            self.current_epoch += 1
            self.save_checkpoint()
    
    def save_checkpoint(self):
        data = {
            "actor_network": self.actor_network,
            "critic_network": self.critic_network,
            "actor_target_network": self.actor_target_network,
            "critic_target_network": self.critic_target_network,
            "current_epoch": self.current_epoch,
            "tot_samples": self.tot_samples,
            "best_success_rate": self.best_success_rate,
            "actor_optim": self.actor_optim,
            "critic_optim": self.critic_optim,
            "actor_sched": self.actor_sched,
            "critic_sched": self.critic_sched,
            "buffer": self.buffer,
            "x_norm": self.x_norm,
            "wandb_run_id": wandb.run.id,
        }
        with open("./checkpoint.pkl", "wb") as f:
            pickle.dump(data, f)

    # pre_process the inputs
    def _preproc_inputs(self, grip, obj, g):
        # concatenate the stuffs
        outs = self.x_norm.normalize(grip, obj, g)
        outs = [torch.tensor(x, dtype=torch.float32) for x in outs]
        if self.args.cuda:
            outs = [x.cuda() for x in outs]
        return outs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi, noise_eps, random_eps):
        action = pi.cpu().numpy()
        # add the gaussian
        action += noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(
            low=-self.env_params['action_max'],
            high=self.env_params['action_max'],
            size=action.shape,
        )
        # choose if use the random actions
        action += np.random.binomial(1, random_eps, (action.shape[0], 1)) * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_grip, mb_obj, mb_ag, mb_g, mb_actions = episode_batch
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        final_t = np.zeros(mb_grip.shape[0], dtype=np.int64)
        final_t[:] = num_transitions
        buffer_temp = {
            'obj': mb_obj, 
            'ag': mb_ag,
            'g': mb_g, 
            'actions': mb_actions, 
            'obs_next': mb_obj[:, 1:, :],
            'ag_next': mb_ag[:, 1:, :],
            'gripper': mb_grip,
            'gripper_next': mb_grip[:, 1:, :],
            'final_t': final_t,
        }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        grip, obj, g = transitions['gripper'], transitions['obj'], transitions['g']
        # pre process the obs and g
        transitions['gripper'], transitions['obj'], transitions['g'] = self._preproc_og(grip, obj, g)
        # update
        self.x_norm.update(transitions['gripper'], transitions['obj'], transitions['g'])
        # recompute the stats
        self.x_norm.recompute_stats()

    def _preproc_og(self, grip, obj, g):
        grip = np.clip(grip, -self.args.clip_obs, self.args.clip_obs)
        obj = np.clip(obj, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return grip, obj, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        grip, grip_next, obj, obj_next, g = transitions['gripper'], transitions['gripper_next'], transitions['obj'], transitions['obj_next'], transitions['g']
        transitions['gripper'], transitions['obs'], transitions['g'] = self._preproc_og(grip, obj, g)
        transitions['gripper_next'], transitions['obs_next'], transitions['g_next'] = self._preproc_og(grip_next, obj_next, g)
        # start to do the update
        inputs_norm = self.x_norm.normalize(transitions['gripper'], transitions['obj'], transitions['g'])
        inputs_next_norm = self.x_norm.normalize(transitions['gripper_next'], transitions['obj_next'], transitions['g_next'])
        # transfer them into the tensor
        inputs_norm_tensor = [torch.tensor(x, dtype=torch.float32) for x in inputs_norm]
        inputs_next_norm_tensor = [torch.tensor(x, dtype=torch.float32) for x in inputs_next_norm]
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = self.args.r_scale * torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = [x.cuda() for x in inputs_norm_tensor]
            inputs_next_norm_tensor = [x.cuda() for x in inputs_next_norm_tensor]
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(*inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(*inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(*inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(*inputs_norm_tensor)
        self.critic_network.eval()
        actor_loss = -self.critic_network(*inputs_norm_tensor, actions_real).mean()
        self.critic_network.train()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        metrics = {
            'loss/actor': actor_loss.detach().cpu().item(),
            'loss/critic': critic_loss.detach().cpu().item(),
        }
        if self.actor_sched is not None:
            self.actor_sched.step()
            metrics['lr/actor'] = self.actor_sched.get_last_lr()[0]
        if self.critic_sched is not None:
            self.critic_sched.step()
            metrics['lr/critic'] = self.critic_sched.get_last_lr()[0]
        return metrics

    # do the evaluation
    def _eval_agent(self):
        self.actor_network.eval()
        results, returns = [], []
        observation = self.eval_env.reset()
        ret = np.zeros(self.args.num_workers)
        while len(results) < self.args.n_test_eps:
            grip, obj = observation['gripper_arr'], observation['object_arr']
            g = observation['desired_goal_arr']
            with torch.no_grad():
                input_tensors = self._preproc_inputs(grip, obj, g)
                pi = self.actor_network(*input_tensors)
                # convert the actions
                actions = pi.detach().cpu().numpy()
            observation_new, rew, done, info = self.eval_env.step(actions)
            ret += rew
            if np.any(done):
                for idx in np.nonzero(done)[0]:
                    results.append(info[idx]['is_success'])
                    returns.append(ret[idx])
                    ret[idx] = 0
            observation = observation_new
        success_rate = np.mean(results)
        ret = np.mean(returns)
        self.actor_network.train()
        return success_rate, ret, self.eval_env.video_recorder.path
