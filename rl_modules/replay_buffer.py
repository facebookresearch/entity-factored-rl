# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on the HeR implementation at https://github.com/TianhongDai/hindsight-experience-replay.

import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obj': np.empty([self.size, self.T + 1, self.env_params['n_objects'], self.env_params['object']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['n_objects'], self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['n_objects'], self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        'gripper': np.empty([self.size, self.T + 1, self.env_params['gripper']]),
                        'final_t': np.empty(self.size, dtype=np.int64),
                        }

    # store the episode
    def store_episode(self, episode_batch):
        mb_grip, mb_obj, mb_ag, mb_g, mb_actions = episode_batch
        T = mb_actions.shape[1]
        batch_size = mb_grip.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['gripper'][idxs, :T+1] = mb_grip
        self.buffers['obj'][idxs, :T+1] = mb_obj
        self.buffers['ag'][idxs, :T+1] = mb_ag
        self.buffers['g'][idxs, :T] = mb_g
        self.buffers['actions'][idxs, :T] = mb_actions
        self.buffers['final_t'][idxs] = T
        self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['gripper_next']  = temp_buffers['gripper'][:, 1:, :]
        temp_buffers['obj_next'] = temp_buffers['obj'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx