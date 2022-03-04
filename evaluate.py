# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import argparse
import pickle
import torch
import gym
import envs, gym_fetch_stack
import numpy as np
import wandb


# pre process the inputs
def preproc_inputs(grip, obj, g, x_norm):
    # concatenate the stuffs
    outs = x_norm.normalize(grip, obj, g)
    return [torch.tensor(x, dtype=torch.float32) for x in outs]


@torch.no_grad()
def main(args):
    api = wandb.Api()
    if args.run_tag:
        assert len(args.run_path) == 1
        runs = api.runs(path=args.run_path[0], filters={"tags": {"$in": [args.run_tag]}})
    else:
        runs = []
        for run_path in args.run_path:
            runs.append(api.run(run_path))
    # load the model param
    for run in runs:
        if run._state != "finished":
            print("Run not finished, skipping.")
            continue
        try:
            run.file('models/best.pt').download(root='/tmp', replace=True)
            x_norm, actor_network = torch.load('/tmp/models/best.pt', map_location=lambda storage, loc: storage)
        except:
            run.file('models/latest.pt').download(root='/tmp', replace=True)
            x_norm, actor_network = torch.load('/tmp/models/latest.pt', map_location=lambda storage, loc: storage)
        actor_network.eval()
        print(f"Evaluating {run.config['actor']['_target_']} in {args.env_name}")
        for env_name in args.env_name:
            if not args.overwrite and f"eval_return/{env_name}" in run.summary:
                print(f"Run {run.id} already has eval for {env_name}.")
                continue
            env = gym.make(env_name)
            results, rets, timesteps = [], [], []
            # start to do the demo
            for ep_num in range(args.num_eps):
                done, ret, t = False, 0, 0
                observation = env.reset()
                while not done:
                    grip, obj = observation['gripper_arr'][None], observation['object_arr'][None]
                    g = observation['desired_goal_arr'][None]
                    inputs = preproc_inputs(grip, obj, g, x_norm)
                    with torch.no_grad():
                        pi = actor_network(*inputs)
                    action = pi.detach().numpy().squeeze()
                    # put actions into the environment
                    observation_new, reward, done, info = env.step(action)
                    observation = observation_new
                    ret += reward
                    done = done or info['is_success']
                    t += 1
                results.append(info['is_success'])
                rets.append(ret)
                timesteps.append(t)
            print(f"{env_name} avg SR ({len(results)} eps): {np.mean(results):0.4f}.")
            run.summary[f"eval_success/{env_name}"] = np.mean(results)
            run.summary[f"eval_return/{env_name}"] = np.mean(rets)
            run.summary[f"eval_length/{env_name}"] = np.mean(timesteps)
            run.summary.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, nargs='+')
    parser.add_argument('--run_tag', type=str)
    parser.add_argument('--run_path', type=str, nargs='+')
    parser.add_argument('--num_eps', type=int, default=200)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(args)