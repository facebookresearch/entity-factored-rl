# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import torch
import gym
import envs, gym_fetch_stack
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
import wandb


# pre process the inputs
def preproc_inputs(grip, obj, g, x_norm):
    # concatenate the stuffs
    outs = x_norm.normalize(grip, obj, g)
    return [torch.tensor(x, dtype=torch.float32) for x in outs]


@torch.no_grad()
def main(args):
    if not os.path.exists('output_videos'):
        os.mkdir('output_videos')
    api = wandb.Api()
    runs = []
    for run_id in args.run_id:
        runs.append(api.run(os.path.join('ayzhong/fetch-her', run_id)))

    # load the model param
    for run in runs:
        run.file('models/best.pt').download(root='/tmp', replace=True)
        x_norm, actor_network = torch.load('/tmp/models/best.pt', map_location=lambda storage, loc: storage)
        actor_network.eval()
        # create the environment
        print(args.env_name)
        for env_name in args.env_name:
            env = gym.make(env_name)
            actor_cls = run.config['actor']['_target_'].split('.')[-1]
            dir = f'./output_videos/{env_name}/{actor_cls}/'
            os.makedirs(dir)
            for i in range(10):
                rec = VideoRecorder(env, base_path=os.path.join(dir, f'vid{i}'))
                observation = env.reset()
                # start to do the demo
                t, solved, solved_step = 0, False, 0
                while t < env._max_episode_steps:
                    if solved and t > solved_step + 10:
                        break
                    rec.capture_frame()
                    grip, obj = observation['gripper_arr'][None], observation['object_arr'][None]
                    g = observation['desired_goal_arr'][None]
                    # env.render()
                    inputs = preproc_inputs(grip, obj, g, x_norm)
                    with torch.no_grad():
                        pi = actor_network(*inputs)
                    action = pi.detach().numpy().squeeze()
                    # put actions into the environment
                    observation_new, reward, done, info = env.step(action)
                    if not solved and info['is_success']:
                        solved_step = t
                    solved = solved or info['is_success']
                    observation = observation_new
                    t += 1
                rec.capture_frame()
                rec.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, nargs='+')
    parser.add_argument('--run_id', type=str, nargs='+')
    args = parser.parse_args()
    main(args)