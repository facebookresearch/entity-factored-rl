# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
import hydra
from omegaconf import OmegaConf
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import wandb
import gym
import envs

from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from rl_modules.normalizer import ArrayNormalizer
from train import get_env_params


def make_env(name):
    def helper():
        import envs
        return gym.make(name)
    return SubprocVecEnv([helper for i in range(16)], start_method="spawn")


def load_data(data_path, x_norm):
    data = np.load(data_path)
    grip, obj, goal, action, success = data["grip"][:, :-1], data["obj"][:, :-1], data["g"], data["action"], data["success"]
    succ_grip, succ_obj, succ_goal, succ_action = [], [], [], []
    print("Filtering data.")
    for ep_idx in range(grip.shape[0]):
        if np.sum(success[ep_idx]) > 0:
            first_succ = np.argmax(success[ep_idx])
            succ_grip.append(grip[ep_idx, :first_succ + 1])
            succ_obj.append(obj[ep_idx, :first_succ + 1])
            succ_goal.append(goal[ep_idx, :first_succ + 1])
            succ_action.append(action[ep_idx, :first_succ + 1])
    succ_grip = np.concatenate(succ_grip, 0)
    succ_obj = np.concatenate(succ_obj, 0)
    succ_goal = np.concatenate(succ_goal, 0)
    succ_action = np.concatenate(succ_action, 0)
    x_norm.update(succ_grip, succ_obj, succ_goal)
    x_norm.recompute_stats()
    succ_grip = torch.from_numpy(succ_grip).float()
    succ_obj = torch.from_numpy(succ_obj).float()
    succ_goal = torch.from_numpy(succ_goal).float()
    succ_action = torch.from_numpy(succ_action).float()
    print(f"Finished filtering data. Total transitions: {succ_grip.shape[0]}.")
    return TensorDataset(succ_grip, succ_obj, succ_goal, succ_action)


def get_opt_sched(policy, args):
    if args.lr_sched == "linear_warmup":
        opt = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.LambdaLR(opt, lambda t: min((t+1) / args.warmup_steps, 1))
    else:
        opt = optim.Adam(policy.parameters(), lr=args.lr)
        scheduler = None
    return opt, scheduler


@torch.no_grad()
def evaluate(env_name, policy, x_norm, device, folder):
    env = make_env(env_name)
    # Not necessarily the same as the training env's max_episode_steps
    max_steps = env.get_attr("_max_episode_steps")[0]
    env = VecVideoRecorder(env, folder, lambda i: i < max_steps, max_steps, name_prefix=env_name)
    def predict_act(observation):
        inp = x_norm.normalize(
            observation["gripper_arr"],
            observation["object_arr"],
            observation["desired_goal_arr"]
        )
        inp = [torch.from_numpy(x).float().to(device) for x in inp]
        return policy(*inp).cpu().numpy()
    observation = env.reset()
    results, step_results = [], []
    # start to do the demo
    while len(results) < 100:
        # env.render()
        action = predict_act(observation)
        # put actions into the environment
        observation_new, reward, done, info = env.step(action)
        if np.any(done):
            for idx in np.nonzero(done)[0]:
                results.append(info[idx]['is_success'])
                step_results.append(info[idx]['step_success'])
        observation = observation_new
    metrics = {
        "success_rate": np.mean(results),
        "step_success_rate": np.mean(step_results),
        "vids/0": wandb.Video(env.video_recorder.path, format="mp4"),
    }
    env.close()
    return metrics


def train(args):
    wandb.init(project="fetch-bc", entity="ayzhong", dir=hydra.utils.get_original_cwd())
    wandb.config.update(OmegaConf.to_container(args, resolve=True))

    device = torch.device(args.device)
    print(f"Using device {device}.")

    train_env = gym.make(args.env_name)
    env_params = get_env_params(train_env)
    del train_env
    x_norm = ArrayNormalizer(env_params, default_clip_range=args.clip_range)

    model_dir = os.path.join(wandb.run.dir, "models")
    os.mkdir(model_dir)

    policy = hydra.utils.instantiate(args.actor, env_params).to(device)
    opt, scheduler = get_opt_sched(policy, args)
    # The path in the config files is relative to the code, but hydra cwd is different.
    data_path = os.path.join(hydra.utils.get_original_cwd(), args.data_path)
    dset = load_data(data_path, x_norm)
    dloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True)

    best_success_rate = -1
    i = 0
    while True:
        for grip, obj, goal, action in dloader:
            inp = x_norm.normalize(grip.numpy(), obj.numpy(), goal.numpy())
            inp = [torch.from_numpy(x).to(device) for x in inp]
            act_predict = policy(*inp)
            loss = F.mse_loss(act_predict, action.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if not i % 500:
                wandb.log({
                    "loss/train": loss.detach().cpu().item(),
                    "lr": opt.param_groups[0]['lr'],
                }, step=i)
            if i % 10000 == 0 or i == args.n_steps - 1:
                save_data = [x_norm, policy]
                torch.save(save_data, os.path.join(model_dir, "latest.pt"))
                start_time = time.time()
                vid_dir = os.path.join(wandb.run.dir, f"./vids_{i}")
                if not os.path.exists(vid_dir):
                    os.mkdir(vid_dir)
                sr_metrics = {}
                policy.eval()
                metrics = evaluate(args.env_name, policy, x_norm, device, vid_dir)
                sr_metrics.update(metrics)
                if metrics["success_rate"] >= best_success_rate or i == 0:
                    best_success_rate = metrics["success_rate"]
                    torch.save(save_data, os.path.join(model_dir, "best.pt"))
                policy.train()
                wandb.log(sr_metrics, step=i)
                print(f"Eval in {time.time() - start_time} seconds.")
            i += 1
            if i == args.n_steps:
                return