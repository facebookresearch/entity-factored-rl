# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import torch
import numpy as np
import gym

import envs
from evaluate import preproc_inputs


ENV_2_CKPT_RL = {
    'Fetch3Push-v1': './weights/3p-3uhwocsy.pt',
    'Fetch3Switch-v1': './weights/3s-2mb98s47.pt',
    'Fetch2Switch2Push-v1': './weights/2s2p-338rpvyu.pt',
}


@torch.no_grad()
def main(args):
    if args.chain:
        if args.env in ['FetchStack2Stage3-v1', 'FetchStack2StitchOnlyStack-v1', 'FetchStack2Stage1-v1']:
            model_paths = {
                'push': './weights/stack-1p-1i0ac7fq.pt',
                'stack': './weights/stack-1s-2q3z23zd.pt',
            }
            networks = {}
            for name, model_path in model_paths.items():
                x_norm, actor_network = torch.load(model_path, map_location=lambda storage, loc: storage)
                actor_network.eval()
                networks[name] = (x_norm, actor_network)
            def policy(observation):
                next_obj_idx = observation['next_object_idx']
                if args.env == 'FetchStack2Stage1-v1':
                    name = 'push'
                else:
                    name = 'push' if next_obj_idx == 0 else 'stack'
                grip = observation['gripper_arr']
                obj = observation['object_arr']
                g = observation['desired_goal_arr']
                x_norm, actor_network = networks[name]
                inputs = preproc_inputs(grip, obj, g, x_norm)
                pi = actor_network(*inputs)
                return pi.numpy().squeeze()
        else:
            model_paths = {}
            if 'Push' in args.env:
                model_paths['push'] = './weights/1p-3n2mu1hy.pt'
            if 'Switch' in args.env:
                model_paths['switch'] = './weights/1s-1gde5bpj.pt'
            networks = {}
            for name, model_path in model_paths.items():
                x_norm, actor_network = torch.load(model_path, map_location=lambda storage, loc: storage)
                actor_network.eval()
                networks[name] = (x_norm, actor_network)
            def policy(observation):
                next_obj_idx = observation['next_object_idx']
                grip = observation['gripper_arr'][None]
                obj = observation['object_arr'][next_obj_idx][None]
                obj_type = np.squeeze(obj[..., -1])
                name = 'switch' if obj_type > 0 else 'push'
                g = observation['desired_goal_arr'][next_obj_idx][None]
                obj, g = obj[None], g[None]
                x_norm, actor_network = networks[name]
                inputs = preproc_inputs(grip, obj, g, x_norm)
                pi = actor_network(*inputs)
                return pi.numpy().squeeze()
    else:
        model_path = ENV_2_CKPT_RL[args.env]
        x_norm, actor_network = torch.load(model_path, map_location=lambda storage, loc: storage)
        actor_network.eval()
        def policy(observation):
            grip = observation['gripper_arr'][None]
            obj, g = observation['object_arr'], observation['desired_goal_arr']
            obj, g = obj[None], g[None]
            inputs = preproc_inputs(grip, obj, g, x_norm)
            pi = actor_network(*inputs)
            return pi.numpy().squeeze()
    print(f"Collecting demos in {args.env}.")
    env = gym.make(args.env)
    observation = env.reset()
    if args.render:
        env = gym.wrappers.Monitor(env, "./demo_videos", force=True)
    observation = env.reset()
    successes = []
    ret_arr, solved_t_arr = [], []
    grip_arr, obj_arr, act_arr, ag_arr, goal_arr, success_arr = [], [], [], [], [], []
    for i in range(args.num_eps):
        # start to do the demo
        ep_grip, ep_obj, ep_act, ep_ag, ep_g, ep_success = [], [], [], [], [], []
        observation = env.reset()
        ret, solved_t = 0, -1
        for t in range(env._max_episode_steps):
            action = policy(observation)
            # put actions into the environment
            observation_new, rew, done, info = env.step(action)
            ep_grip.append(observation["gripper_arr"].copy())
            ep_obj.append(observation["object_arr"].copy())
            ep_ag.append(observation["achieved_goal_arr"].copy())
            ep_g.append(observation["desired_goal_arr"].copy())
            ep_act.append(action.copy())
            ep_success.append(info["is_success"])
            observation = observation_new
            if solved_t < 0:
                ret += rew
            if info['is_success'] and solved_t < 0:
                solved_t = t
        ep_grip.append(observation["gripper_arr"].copy())
        ep_obj.append(observation["object_arr"].copy())
        ep_ag.append(observation["achieved_goal_arr"].copy())
        print(f'episode {i}, is success: {info["is_success"]}, finished t: {solved_t}')
        successes.append(info['is_success'])
        grip_arr.append(np.stack(ep_grip))
        obj_arr.append(np.stack(ep_obj))
        act_arr.append(np.stack(ep_act))
        ag_arr.append(np.stack(ep_ag))
        goal_arr.append(np.stack(ep_g))
        success_arr.append(np.stack(ep_success))
        ret_arr.append(ret)
        solved_t_arr.append(env._max_episode_steps if solved_t < 0 else solved_t)
    grip_arr = np.stack(grip_arr)
    obj_arr = np.stack(obj_arr)
    act_arr = np.stack(act_arr)
    ag_arr = np.stack(ag_arr)
    goal_arr = np.stack(goal_arr)
    success_arr = np.stack(success_arr)
    ret_arr = np.array(ret_arr)
    solved_t_arr = np.array(solved_t_arr)
    name = "init_trajs" if args.chain else "rl_expert"
    np.savez(
        f"./data/{name}_{args.env}_{args.num_eps}.npz",
        grip=grip_arr,
        obj=obj_arr,
        action=act_arr,
        ag=ag_arr,
        g=goal_arr,
        success=success_arr,
        ret_arr=ret_arr,
        solved_t_arr=solved_t_arr,
    )
    print(np.mean(successes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, default="Fetch3Push-v1")
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--num_eps", type=int, default=1500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    main(args)
