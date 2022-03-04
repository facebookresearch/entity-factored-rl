# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from gym.envs.registration import registry, register, make, spec
    
for reward_type in ["sparse", "dense", "step", "hybrid", "eff"]:
    suffix = {
        "sparse": "",
        "dense": "Dense",
        "step": "Step",
        "hybrid": "Hybrid",
        "eff": "Eff",
    }[reward_type]
    kwargs = {
        "reward_type": reward_type,
    }

    for i in range(1, 4):

        register(
            id=f"Fetch{i}Switch{i}Push{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchMPushEnv",
            kwargs={"num_objects": i, "num_switches": i, **kwargs},
            max_episode_steps=70 * i,
        )

        register(
            id=f"Fetch{i}Switch{i}PushCollide{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchMPushEnv",
            kwargs={"num_objects": i, "num_switches": i, 'collisions': True, **kwargs},
            max_episode_steps=70 * i,
        )

        register(
            id=f"Fetch{i}SwitchOr{i}Push{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchMPushEnv",
            kwargs={"num_objects": i, "num_switches": i, "push_switch_exclusive": "random", **kwargs},
            max_episode_steps=50 * i,
        )

        register(
            id=f"Fetch{i}SwitchOr{i}PushCollide{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchMPushEnv",
            kwargs={"num_objects": i, "num_switches": i, "collisions": True, "push_switch_exclusive": "random", **kwargs},
            max_episode_steps=50 * i,
        )

        register(
            id=f"Fetch{i}SwitchOr{i}PushOnlyCube{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchMPushEnv",
            kwargs={"num_objects": i, "num_switches": i, "push_switch_exclusive": "cube", **kwargs},
            max_episode_steps=50 * i,
        )

        register(
            id=f"Fetch{i}SwitchOr{i}PushOnlySwitch{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchMPushEnv",
            kwargs={"num_objects": i, "num_switches": i, "push_switch_exclusive": "switch", **kwargs},
            max_episode_steps=50 * i,
        )

        register(
            id=f"Fetch{i}SwitchOr{i}PushOnlyCubeCollide{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchMPushEnv",
            kwargs={"num_objects": i, "num_switches": i, "push_switch_exclusive": "cube", "collisions": True, **kwargs},
            max_episode_steps=50 * i,
        )

        register(
            id=f"Fetch{i}SwitchOr{i}PushOnlySwitchCollide{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchMPushEnv",
            kwargs={"num_objects": i, "num_switches": i, "push_switch_exclusive": "switch", "collisions": True, **kwargs},
            max_episode_steps=50 * i,
        )

    for num_pairs in range(1, 7):
        max_steps = 50 * num_pairs

        register(
            id=f"Fetch{num_pairs}Push{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNPushEnv",
            kwargs={"num_objects": num_pairs, **kwargs},
            max_episode_steps=max_steps,
        )

        register(
            id=f"Fetch{num_pairs}PushCollide{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNPushEnv",
            kwargs={"num_objects": num_pairs, "collisions": True, **kwargs},
            max_episode_steps=max_steps,
        )

    for num_switches in range(1, 7):
        max_steps = 20 * num_switches
        register(
            id=f"Fetch{num_switches}Switch{suffix}-v1",
            entry_point="envs.fetch_push_multi:FetchNSwitchEnv",
            kwargs={"num_switches": num_switches, **kwargs},
            max_episode_steps=max_steps,
        )