# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on the HeR implementation at https://github.com/TianhongDai/hindsight-experience-replay.

import os
import pickle
import hydra
import wandb
from omegaconf import OmegaConf
import gym
import mujoco_py
from rl_modules.ddpg_agent import ddpg_agent
from vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder, vec_normalize
import envs, gym_fetch_stack


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {
        'gripper': obs['gripper_arr'].shape[-1],
        'goal': obs['desired_goal_arr'].shape[-1],
        'action': env.action_space.shape[0],
        'action_max': env.action_space.high[0],
        'object': obs['object_arr'].shape[-1],
        'n_objects': obs['object_arr'].shape[0],
    }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    # get the environment parameters
    test_env = gym.make(args.env_name)
    env_params = get_env_params(test_env)
    test_env.close()
    # create the ddpg_agent
    def make_env():
        import envs, gym_fetch_stack  # needed when using start_method="spawn"
        return gym.make(args.env_name)
    env = SubprocVecEnv([make_env for i in range(args.num_workers)], start_method="spawn")
    if args.norm_reward:
        env = vec_normalize.VecNormalize(env, norm_obs=False, norm_reward=True, gamma=args.gamma)
    eval_env = SubprocVecEnv([make_env for i in range(args.num_workers)], start_method="spawn", auto_reset=True)
    eval_env = VecVideoRecorder(eval_env, "eval_vids", lambda i: i < env_params['max_timesteps'], env_params['max_timesteps'])

    ckpt_data, wid = None, None
    ckpt_path = "./checkpoint.pkl"
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            print(f"Loading data from {ckpt_path}.")
            ckpt_data = pickle.load(f)
            wid = ckpt_data["wandb_run_id"]
    # create the ddpg agent to interact with the environment 
    wandb.init(project='fetch-her', entity='ayzhong', id=wid, resume="allow", dir=hydra.utils.get_original_cwd())
    if wid is None:
        wandb.config.update(OmegaConf.to_container(args, resolve=True))
    ddpg_trainer = ddpg_agent(args, env, eval_env, env_params, test_env.compute_reward, ckpt_data=ckpt_data)
    ddpg_trainer.learn()
