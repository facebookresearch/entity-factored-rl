# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on the HeR implementation at https://github.com/TianhongDai/hindsight-experience-replay.

import numpy as np

class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        self.total_sum += v.sum(axis=0)
        self.total_sumsq += (np.square(v)).sum(axis=0)
        self.total_count[0] += v.shape[0]

    def recompute_stats(self):
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)


class ArrayNormalizer:
    def __init__(self, env_params, default_clip_range=np.inf):
        self.gripper_norm = normalizer(env_params['gripper'], default_clip_range=default_clip_range)
        self.objects_norm = normalizer(env_params['object'], default_clip_range=default_clip_range)
        self.goal_norm = normalizer(env_params['goal'], default_clip_range=default_clip_range)

    def update(self, gripper, objects, goal):
        self.gripper_norm.update(gripper)
        self.objects_norm.update(objects)
        self.goal_norm.update(goal)

    def recompute_stats(self):
        self.gripper_norm.recompute_stats()
        self.objects_norm.recompute_stats()
        self.goal_norm.recompute_stats()

    # normalize the observation
    def normalize(self, gripper, objects, goal, clip_range=None):
        gripper_norm = self.gripper_norm.normalize(gripper, clip_range)
        objects_norm = self.objects_norm.normalize(objects, clip_range)
        goal_norm = self.goal_norm.normalize(goal, clip_range)
        return gripper_norm, objects_norm, goal_norm