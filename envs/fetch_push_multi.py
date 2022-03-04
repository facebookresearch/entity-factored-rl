# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on the Fetch robotics environment implementation at https://github.com/openai/gym.

import os
import pathlib
import numpy as np

from gym import utils as gym_utils
from gym.envs.robotics import rotations, robot_env, utils
from gym import spaces

from mujoco_py.generated import const


PARENT_DIR = pathlib.Path(__file__).parent.resolve()


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchMultiEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        block_gripper,
        num_objects,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        initial_qpos,
        reward_type,
        num_switches=0,
        push_switch_exclusive=None,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            num_objects (int): number of objects in the environment
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.num_objects = num_objects
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.num_switches = num_switches
        # If True, in each episode only the cubes can be away from their goals (or the switches)
        # But not both.
        self.push_switch_exclusive = push_switch_exclusive

        super(FetchMultiEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
        )

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                desired_goal_arr=spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal_arr"].shape, dtype="float32"
                ),
                achieved_goal_arr=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal_arr"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
                gripper_arr=spaces.Box(
                    -np.inf, np.inf, shape=obs["gripper_arr"].shape, dtype="float32"
                ),
                object_arr=spaces.Box(
                    -np.inf, np.inf, shape=obs["object_arr"].shape, dtype="float32"
                ),
            )
        )

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        sparse_reward = -np.any(d > self.distance_threshold, axis=-1).astype(np.float32)
        if self.reward_type == "sparse":
            return sparse_reward
        elif self.reward_type == "step":
            return -np.mean(d > self.distance_threshold, axis=-1)
        elif self.reward_type == "hybrid":
            return sparse_reward - .2 * np.mean(d, axis=-1)
        elif self.reward_type == "eff":
            grip_pos = info["gripper_arr"][..., :3]
            obj_d = np.linalg.norm(grip_pos[..., None, :] - achieved_goal, axis=-1)
            unsolved = d > self.distance_threshold
            grip_rew = 0
            if unsolved.sum() > 0:
                grip_rew = obj_d[unsolved].mean(axis=-1)
            return sparse_reward - 5 * np.mean(d, axis=-1) - grip_rew
        elif self.reward_type == "dense":
            dense_cost = d.copy()
            # Clip switch cost to 0.7, i.e. the cost when the switch is center. So you are
            # not penalized for pushing the switch in the wrong direction. The 0.6 scaling
            # is heuristic, for balancing with cube costs.
            dense_cost[self.num_objects:] = .6 * np.clip(dense_cost[self.num_objects:], 0., 0.7)
            return -np.mean(dense_cost, axis=-1)
        else:
            raise ValueError("Unrecognized reward type.")

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_data = np.zeros(0)
        if self.num_objects > 0 or self.num_switches > 0:
            object_pos_all, object_rel_pos_all = [], []
            object_rot_all, object_velp_all, object_velr_all = [], [], []
            object_ids_all = []  # 0 for cubes, 1 for switches.
            for i in range(self.num_objects):
                object_name = "object" + str(i)
                object_pos = self.sim.data.get_site_xpos(object_name)
                # rotations
                object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(object_name))
                # velocities
                object_velp = self.sim.data.get_site_xvelp(object_name) * dt
                object_velr = self.sim.data.get_site_xvelr(object_name) * dt
                # gripper state
                object_rel_pos = object_pos - grip_pos
                object_velp -= grip_velp

                object_pos_all.append(object_pos.ravel())
                object_rel_pos_all.append(object_rel_pos.ravel())
                object_rot_all.append(object_rot.ravel())
                object_velp_all.append(object_velp.ravel())
                object_velr_all.append(object_velr.ravel())
                object_ids_all.append(0.0)
            for i in range(self.num_switches):
                switch_name = "lightswitchbase" + str(i)
                # Switch pos is just the joint angle, padded.
                switch_pos = np.zeros(3)
                switch_pos[0] = self.sim.data.get_joint_qpos(f"lightswitchroot{i}:joint")
                switch_xpos = self.sim.data.get_site_xpos(switch_name)
                switch_rel_xpos = switch_xpos - grip_pos
                switch_rot = rotations.mat2euler(self.sim.data.get_site_xmat(switch_name))
                object_pos_all.append(switch_pos.ravel())
                object_rel_pos_all.append(switch_rel_xpos.ravel())
                object_rot_all.append(switch_rot.ravel())
                # For switch we give base xpos where the object gives velp.
                object_velp_all.append(switch_xpos)
                object_velr_all.append(np.zeros(3))
                object_ids_all.append(1.0)
            object_pos_all = np.stack(object_pos_all)
            object_rel_pos_all = np.stack(object_rel_pos_all)
            object_rot_all = np.stack(object_rot_all)
            object_velp_all = np.stack(object_velp_all)
            object_velr_all = np.stack(object_velr_all)
            object_ids_all = np.array(object_ids_all)[:, None]
            object_data = np.concatenate([
                object_pos_all,
                object_rel_pos_all,
                object_rot_all,
                object_velp_all,
                object_velr_all,
                object_ids_all,
            ], axis=-1)
        gripper_state = robot_qpos[-2:]
        # change to a scalar if the gripper is made symmetric
        gripper_vel = robot_qvel[-2:] * dt

        if self.num_objects == 0 and self.num_switches == 0:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = object_pos_all

        gripper_data = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel], axis=0)
        observation = np.concatenate((gripper_data, object_data.ravel()), axis=0)
        desired_goal = self.goal.copy()
        obs_dict = {
            "gripper_arr": gripper_data,
            "object_arr": object_data,
            "achieved_goal_arr": achieved_goal,
            "desired_goal_arr": desired_goal,
            # Flattened observations
            "observation": observation,
            "achieved_goal": achieved_goal.ravel(),
            "desired_goal": desired_goal.ravel(),
        }

        goal_dists = goal_distance(achieved_goal, self.goal)
        next_idx = np.argmax(goal_dists > self.distance_threshold)
        obs_dict["next_object_idx"] = next_idx
        return obs_dict

    def _viewer_setup(self):
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.fixedcamid = 3

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        if self.num_objects > 0:
            for i in range(self.num_objects):
                target_name = "target" + str(i)
                site_id = self.sim.model.site_name2id(target_name)
                self.sim.model.site_pos[site_id] = self.goal[i] - sites_offset[0]
        elif self.num_switches == 0:
            site_id = self.sim.model.site_name2id("target0")
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.num_objects > 0 or self.num_switches > 0:
            prev_objects = []
            for i in range(self.num_objects):
                object_xpos = self.initial_gripper_xpos[:2]
                while True:
                    dist_gripper = np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2])
                    close_to_prev = False
                    for obj in prev_objects:
                        close_to_prev = close_to_prev or (np.linalg.norm(object_xpos - obj) < 0.1)
                    if dist_gripper < 0.1 or close_to_prev:
                        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                            -self.obj_range, self.obj_range, size=2
                        )
                    else:
                        prev_objects.append(object_xpos)
                        break
                joint_name = "object" + str(i) + ":joint"
                object_qpos = self.sim.data.get_joint_qpos(joint_name)
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos(joint_name, object_qpos)
            for i in range(self.num_switches):
                object_xpos = self.initial_gripper_xpos[:2]
                while True:
                    dist_gripper = np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2])
                    close_to_prev = False
                    for obj in prev_objects:
                        close_to_prev = close_to_prev or (np.linalg.norm(object_xpos - obj) < 0.1)
                    if dist_gripper < 0.1 or close_to_prev:
                        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                            -self.obj_range, self.obj_range, size=2
                        )
                    else:
                        prev_objects.append(object_xpos)
                        break
                switch_id = self.sim.model.body_name2id(f"lightswitchbase{i}")
                self.sim.model.body_pos[switch_id][:2] = object_xpos
        self.sim.forward()
        return True

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["achieved_goal_arr"], obs["desired_goal_arr"]),
            "step_success": self._step_success(obs["achieved_goal_arr"], obs["desired_goal_arr"]),
            "gripper_arr": obs["gripper_arr"],
        }
        done = int(info["is_success"])  # Early stopping.
        reward = self.compute_reward(obs["achieved_goal_arr"], obs["desired_goal_arr"], info)
        return obs, reward, done, info

    def _sample_goal(self):
        goals = np.zeros((0, 3))
        if self.num_objects > 0 or self.num_switches > 0:
            # Record the position of the switches--the object goals shouldn't be too close to them.
            switch_xypos = np.zeros((0, 2))
            for i in range(self.num_switches):
                switch_id = self.sim.model.body_name2id(f"lightswitchbase{i}")
                switch_xypos = np.concatenate((switch_xypos, self.sim.model.body_pos[switch_id][:2][None]), axis=0)
            for i in range(self.num_objects):
                while True:
                    goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                        -self.target_range, self.target_range, size=3
                    )
                    goal += self.target_offset
                    goal[2] = self.height_offset
                    if self.target_in_the_air and self.np_random.uniform() < 0.5:
                        goal[2] += self.np_random.uniform(0, 0.45)
                    sep_goal = (not i) or np.min(np.linalg.norm(goals - goal[None], axis=-1)) > 0.1
                    sep_switch = (not self.num_switches) or np.min(np.linalg.norm(switch_xypos - goal[:2][None], axis=-1)) > 0.1
                    if sep_goal and sep_switch:
                        break
                goals = np.concatenate((goals, goal[None]), 0)
            for i in range(self.num_switches):
                goal = np.array([self.np_random.choice([-0.7, 0.7]), 0, 0])
                goals = np.concatenate((goals, goal[None]), 0)
            if self.push_switch_exclusive is not None:
                if self.push_switch_exclusive == "random":
                    use_cubes = self.np_random.uniform() < 0.5
                else:
                    assert self.push_switch_exclusive in ["cube", "switch"]
                    use_cubes = self.push_switch_exclusive == "cube"
                if use_cubes:  # put the switches to their goal
                    for i in range(self.num_objects):
                        joint_name = "object" + str(i) + ":joint"
                        object_qpos = self.sim.data.get_joint_qpos(joint_name)
                        assert object_qpos.shape == (7,)
                        object_qpos[:2] = goals[i][:2]
                        self.sim.data.set_joint_qpos(joint_name, object_qpos)
                else:  # put the cubes to their goal
                    for i in range(self.num_switches):
                        joint_name = f"lightswitchroot{i}:joint"
                        self.sim.data.set_joint_qpos(joint_name, goals[i + self.num_objects][0])
                self.sim.forward()
        else:
            goals = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        return goals.copy()

    def _step_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return np.mean(d < self.distance_threshold, axis=-1)

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return np.all(d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self.sim.data.get_site_xpos("robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        if self.num_objects > 0:
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]

    def render(self, mode="human", width=500, height=500):
        return super(FetchMultiEnv, self).render(mode, width, height)


PUSH_OBJ_INIT = {
    "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
    "object1:joint": [1.30, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
    "object2:joint": [1.35, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
    "object3:joint": [1.25, 0.60, 0.4, 1.0, 0.0, 0.0, 0.0],
    "object4:joint": [1.30, 0.60, 0.4, 1.0, 0.0, 0.0, 0.0],
    "object5:joint": [1.35, 0.60, 0.4, 1.0, 0.0, 0.0, 0.0],
}


class FetchNPushEnv(FetchMultiEnv, gym_utils.EzPickle):
    def __init__(self, reward_type="sparse", num_objects=1, collisions=False, **kwargs):
        if collisions:
            model_xml_path = os.path.join(PARENT_DIR, f"assets/fetch/push{num_objects}_collide.xml")
        else:
            model_xml_path = os.path.join(PARENT_DIR, f"assets/fetch/push{num_objects}.xml")
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        for i in range(num_objects):
            name = f"object{i}:joint"
            initial_qpos[name] = PUSH_OBJ_INIT[name]
        FetchMultiEnv.__init__(
            self,
            model_xml_path,
            num_objects=num_objects,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        gym_utils.EzPickle.__init__(self, reward_type=reward_type)


class FetchNSwitchEnv(FetchMultiEnv, gym_utils.EzPickle):
    def __init__(self, reward_type="sparse", num_switches=1, **kwargs):
        model_xml_path = os.path.join(PARENT_DIR, f"assets/fetch/switch{num_switches}.xml")
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        FetchMultiEnv.__init__(
            self,
            model_xml_path,
            num_objects=0,
            num_switches=num_switches,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        gym_utils.EzPickle.__init__(self, reward_type=reward_type)


class FetchNSwitchMPushEnv(FetchMultiEnv, gym_utils.EzPickle):
    def __init__(self, reward_type="sparse", num_objects=1, num_switches=1, collisions=False, **kwargs):
        if collisions:
            model_xml_path = os.path.join(PARENT_DIR, f"assets/fetch/switch{num_switches}push{num_objects}_collide.xml")
        else:
            model_xml_path = os.path.join(PARENT_DIR, f"assets/fetch/switch{num_switches}push{num_objects}.xml")
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        for i in range(num_objects):
            name = f"object{i}:joint"
            initial_qpos[name] = PUSH_OBJ_INIT[name]
        FetchMultiEnv.__init__(
            self,
            model_xml_path,
            num_objects=num_objects,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            num_switches=num_switches,
            **kwargs,
        )
        gym_utils.EzPickle.__init__(self, reward_type=reward_type)