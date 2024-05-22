import numpy
import json

import torch
from isaacgym.torch_utils import *
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import LeggedRobotFFTAI
from legged_gym.utils.math import quat_apply_yaw
from legged_gym.utils.helpers import class_to_dict

from .gr1t1_config import GR1T1Cfg


class GR1T1(LeggedRobotFFTAI):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.swing_feet_height_target = torch.ones(self.num_envs, 1,
                                                   dtype=torch.float, device=self.device, requires_grad=False) \
                                        * self.cfg.rewards.swing_feet_height_target

        # additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs * self.actor_num_output, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs * self.actor_num_output, 3, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs * self.actor_num_output, 3, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.actor_num_output, 3)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.actor_num_output, 3)

    def _init_cfg(self, cfg: GR1T1Cfg):
        super()._init_cfg(cfg)

    def _create_envs_get_indices(self, body_names, env_handle, actor_handle):
        """ Creates a list of indices for different bodies of the robot.
        """
        torso_name = [s for s in body_names if self.cfg.asset.torso_name in s]
        chest_name = [s for s in body_names if self.cfg.asset.chest_name in s]
        forehead_indices = [s for s in body_names if self.cfg.asset.forehead_name in s]

        imu_name = [s for s in body_names if self.cfg.asset.imu_name in s]

        waist_names = [s for s in body_names if self.cfg.asset.waist_name in s]
        head_names = [s for s in body_names if self.cfg.asset.head_name in s]
        thigh_names = [s for s in body_names if self.cfg.asset.thigh_name in s]
        shank_names = [s for s in body_names if self.cfg.asset.shank_name in s]
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        sole_names = [s for s in body_names if self.cfg.asset.sole_name in s]
        upper_arm_names = [s for s in body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in body_names if self.cfg.asset.lower_arm_name in s]
        hand_names = [s for s in body_names if self.cfg.asset.hand_name in s]

        arm_base_names = [s for s in body_names if self.cfg.asset.arm_base_name in s]
        arm_end_names = [s for s in body_names if self.cfg.asset.arm_end_name in s]

        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, torso_name[j])

        self.chest_indices = torch.zeros(len(chest_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(chest_name)):
            self.chest_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, chest_name[j])

        self.forehead_indices = torch.zeros(len(forehead_indices), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(forehead_indices)):
            self.forehead_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, forehead_indices[j])

        self.imu_indices = torch.zeros(len(imu_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(imu_name)):
            self.imu_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, imu_name[j])

        self.waist_indices = torch.zeros(len(waist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(waist_names)):
            self.waist_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, waist_names[j])

        self.head_indices = torch.zeros(len(head_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(head_names)):
            self.head_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, head_names[j])

        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(thigh_names)):
            self.thigh_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, thigh_names[j])

        self.shank_indices = torch.zeros(len(shank_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(shank_names)):
            self.shank_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, shank_names[j])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(feet_names)):
            self.feet_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, feet_names[j])

        self.sole_indices = torch.zeros(len(sole_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(sole_names)):
            self.sole_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, sole_names[j])

        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, upper_arm_names[j])

        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, lower_arm_names[j])

        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(hand_names)):
            self.hand_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, hand_names[j])

        self.arm_base_indices = torch.zeros(len(arm_base_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(arm_base_names)):
            self.arm_base_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, arm_base_names[j])

        self.arm_end_indices = torch.zeros(len(arm_end_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(arm_end_names)):
            self.arm_end_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, arm_end_names[j])

        print("self.torso_indices: " + str(self.torso_indices))
        print("self.chest_indices: " + str(self.chest_indices))
        print("self.forehead_indices: " + str(self.forehead_indices))

        print("self.imu_indices: " + str(self.imu_indices))

        print("self.waist_indices: " + str(self.waist_indices))
        print("self.head_indices: " + str(self.head_indices))

        print("self.thigh_indices: " + str(self.thigh_indices))
        print("self.shank_indices: " + str(self.shank_indices))
        print("self.feet_indices: " + str(self.feet_indices))
        print("self.sole_indices: " + str(self.sole_indices))

        print("self.upper_arm_indices: " + str(self.upper_arm_indices))
        print("self.lower_arm_indices: " + str(self.lower_arm_indices))
        print("self.hand_indices: " + str(self.hand_indices))

        print("self.arm_base_indices: " + str(self.arm_base_indices))
        print("self.arm_end_indices: " + str(self.arm_end_indices))

    def _init_buffers(self):
        super()._init_buffers()

        # Jason 2023-09-19:
        # change from actor_num_output to num_dof
        self.actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.dof_pos_leg = torch.zeros(self.num_envs, len(self.leg_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_leg = torch.zeros(self.num_envs, len(self.leg_indices), dtype=torch.float, device=self.device, requires_grad=False)

        # commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_heading = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.ones_like(self.commands, dtype=torch.float, device=self.device, requires_grad=False)

        # resample stand command env_ids
        self.env_ids_for_stand_command = list(range(self.num_envs))
        self.env_ids_for_walk_command = list(range(self.num_envs))

        self._init_buffer_orient()
        self._init_buffers_gait_phase()

    def _init_buffers_joint_indices(self):

        # get joint indices
        waist_names = self.cfg.asset.waist_name
        waist_yaw_names = self.cfg.asset.waist_yaw_name
        waist_roll_names = self.cfg.asset.waist_roll_name
        waist_pitch_names = self.cfg.asset.waist_pitch_name
        head_names = self.cfg.asset.head_name
        head_roll_names = self.cfg.asset.head_roll_name
        head_pitch_names = self.cfg.asset.head_pitch_name
        hip_names = self.cfg.asset.hip_name
        hip_roll_names = self.cfg.asset.hip_roll_name
        hip_pitch_names = self.cfg.asset.hip_pitch_name
        hip_yaw_names = self.cfg.asset.hip_yaw_name
        knee_names = self.cfg.asset.knee_name
        ankle_names = self.cfg.asset.ankle_name
        ankle_pitch_names = self.cfg.asset.ankle_pitch_name
        ankle_roll_names = self.cfg.asset.ankle_roll_name
        shoulder_names = self.cfg.asset.shoulder_name
        shoulder_pitch_names = self.cfg.asset.shoulder_pitch_name
        shoulder_roll_names = self.cfg.asset.shoulder_roll_name
        shoulder_yaw_names = self.cfg.asset.shoulder_yaw_name
        elbow_names = self.cfg.asset.elbow_name
        wrist_names = self.cfg.asset.wrist_name
        wrist_yaw_names = self.cfg.asset.wrist_yaw_name
        wrist_roll_names = self.cfg.asset.wrist_roll_name
        wrist_pitch_names = self.cfg.asset.wrist_pitch_name

        self.waist_indices = []
        self.waist_yaw_indices = []
        self.waist_roll_indices = []
        self.waist_pitch_indices = []
        self.head_indices = []
        self.head_roll_indices = []
        self.head_pitch_indices = []
        self.hip_indices = []
        self.hip_roll_indices = []
        self.hip_pitch_indices = []
        self.hip_yaw_indices = []
        self.knee_indices = []
        self.ankle_indices = []
        self.ankle_pitch_indices = []
        self.ankle_roll_indices = []
        self.shoulder_indices = []
        self.shoulder_pitch_indices = []
        self.shoulder_roll_indices = []
        self.shoulder_yaw_indices = []
        self.elbow_indices = []
        self.wrist_indices = []
        self.wrist_yaw_indices = []
        self.wrist_roll_indices = []
        self.wrist_pitch_indices = []

        self.leg_indices = []
        self.arm_indices = []

        self.left_leg_indices = []
        self.right_leg_indices = []
        self.left_arm_indices = []
        self.right_arm_indices = []

        for i in range(self.num_dof):
            name = self.dof_names[i]

            if waist_names in name:
                self.waist_indices.append(i)

            if waist_yaw_names in name:
                self.waist_yaw_indices.append(i)

            if waist_roll_names in name:
                self.waist_roll_indices.append(i)

            if waist_pitch_names in name:
                self.waist_pitch_indices.append(i)

            if head_names in name:
                self.head_indices.append(i)

            if head_roll_names in name:
                self.head_roll_indices.append(i)

            if head_pitch_names in name:
                self.head_pitch_indices.append(i)

            if hip_names in name:
                self.hip_indices.append(i)
                self.leg_indices.append(i)

            if hip_roll_names in name:
                self.hip_roll_indices.append(i)

            if hip_pitch_names in name:
                self.hip_pitch_indices.append(i)

            if hip_yaw_names in name:
                self.hip_yaw_indices.append(i)

            if knee_names in name:
                self.knee_indices.append(i)
                self.leg_indices.append(i)

            if ankle_names in name:
                self.ankle_indices.append(i)
                self.leg_indices.append(i)

            if ankle_pitch_names in name:
                self.ankle_pitch_indices.append(i)

            if ankle_roll_names in name:
                self.ankle_roll_indices.append(i)

            if shoulder_names in name:
                self.shoulder_indices.append(i)
                self.arm_indices.append(i)

            if shoulder_pitch_names in name:
                self.shoulder_pitch_indices.append(i)

            if shoulder_roll_names in name:
                self.shoulder_roll_indices.append(i)

            if shoulder_yaw_names in name:
                self.shoulder_yaw_indices.append(i)

            if elbow_names in name:
                self.elbow_indices.append(i)
                self.arm_indices.append(i)

            if wrist_names in name:
                self.wrist_indices.append(i)
                self.arm_indices.append(i)

            if wrist_yaw_names in name:
                self.wrist_yaw_indices.append(i)

            if wrist_roll_names in name:
                self.wrist_roll_indices.append(i)

            if wrist_pitch_names in name:
                self.wrist_pitch_indices.append(i)

        print("self.waist_indices: " + str(self.waist_indices))
        print("self.waist_yaw_indices: " + str(self.waist_yaw_indices))
        print("self.waist_roll_indices: " + str(self.waist_roll_indices))
        print("self.waist_pitch_indices: " + str(self.waist_pitch_indices))
        print("self.head_indices: " + str(self.head_indices))
        print("self.head_roll_indices: " + str(self.head_roll_indices))
        print("self.head_pitch_indices: " + str(self.head_pitch_indices))

        print("self.hip_indices: " + str(self.hip_indices))
        print("self.hip_roll_indices: " + str(self.hip_roll_indices))
        print("self.hip_pitch_indices: " + str(self.hip_pitch_indices))
        print("self.hip_yaw_indices: " + str(self.hip_yaw_indices))
        print("self.knee_indices: " + str(self.knee_indices))
        print("self.ankle_indices: " + str(self.ankle_indices))
        print("self.ankle_pitch_indices: " + str(self.ankle_pitch_indices))
        print("self.ankle_roll_indices: " + str(self.ankle_roll_indices))
        print("self.shoulder_indices: " + str(self.shoulder_indices))
        print("self.shoulder_pitch_indices: " + str(self.shoulder_pitch_indices))
        print("self.shoulder_roll_indices: " + str(self.shoulder_roll_indices))
        print("self.shoulder_yaw_indices: " + str(self.shoulder_yaw_indices))
        print("self.elbow_indices: " + str(self.elbow_indices))
        print("self.wrist_indices: " + str(self.wrist_indices))
        print("self.wrist_yaw_indices: " + str(self.wrist_yaw_indices))
        print("self.wrist_roll_indices: " + str(self.wrist_roll_indices))
        print("self.wrist_pitch_indices: " + str(self.wrist_pitch_indices))

        print("self.leg_indices: " + str(self.leg_indices))
        print("self.arm_indices: " + str(self.arm_indices))

        self.left_leg_indices = self.leg_indices[:len(self.leg_indices) // 2]
        self.right_leg_indices = self.leg_indices[len(self.leg_indices) // 2:]
        self.left_arm_indices = self.arm_indices[:len(self.arm_indices) // 2]
        self.right_arm_indices = self.arm_indices[len(self.arm_indices) // 2:]

        print("self.left_leg_indices: " + str(self.left_leg_indices))
        print("self.right_leg_indices: " + str(self.right_leg_indices))
        print("self.left_arm_indices: " + str(self.left_arm_indices))
        print("self.right_arm_indices: " + str(self.right_arm_indices))

    def _init_buffers_measure_heights(self):
        super()._init_buffers_measure_heights()

        # measured height supervisor
        if self.cfg.terrain.measure_heights_supervisor:
            self.height_points_supervisor = self._init_height_points_supervisor()
        self.measured_heights_supervisor = None

    def _init_buffer_orient(self):
        self._calculate_feet_orient()
        self._calculate_imu_orient()

    def _parse_cfg(self):

        print("----------------------------------------")

        super()._parse_cfg()

        # gait cycle ---------------------------------------

        self.ranges_phase_ratio = class_to_dict(self.cfg.commands.ranges_phase_ratio)
        self.ranges_gait_cycle = class_to_dict(self.cfg.commands.ranges_gait_cycle)
        self.ranges_swing_feet_height = class_to_dict(self.cfg.commands.ranges_swing_feet_height)

        # gait cycle ---------------------------------------

        # walk --------------------------------------------

        if self.cfg.commands.resample_command_profiles.__contains__("GR1T1-walk"):
            self.command_ranges_walk = class_to_dict(self.cfg.commands.ranges_walk)
            print("self.command_ranges_walk: \n",
                  json.dumps(self.command_ranges_walk, indent=4, sort_keys=True))

            self.reward_scales_walk = class_to_dict(self.cfg.rewards.scales_walk)
            print("self.reward_scales_walk: \n",
                  json.dumps(self.reward_scales_walk, indent=4, sort_keys=True))

        # walk --------------------------------------------

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        super()._prepare_reward_function()

        # walk --------------------------------------------

        if self.cfg.commands.resample_command_profiles.__contains__("GR1T1-walk"):
            # remove zero scales + multiply non-zero ones by dt
            for key in list(self.reward_scales_walk.keys()):
                scale = self.reward_scales_walk[key]
                if scale == 0:
                    self.reward_scales_walk.pop(key)
                else:
                    self.reward_scales_walk[key] *= self.dt

            # prepare list of functions
            self.reward_functions_walk = []
            self.reward_names_walk = []
            for name, scale in self.reward_scales_walk.items():
                if name == "termination":
                    continue
                self.reward_names_walk.append(name)
                name = '_reward_' + name
                self.reward_functions_walk.append(getattr(self, name))

            # reward episode sums
            # get name in self.reward_scales_walk.keys(), but not in self.reward_scales.keys()
            for name in set(self.reward_scales_walk.keys()) - set(self.reward_scales.keys()):
                self.episode_sums[name] = torch.zeros(self.num_envs,
                                                      dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)

        # walk --------------------------------------------

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0

        # walk --------------------------------------------

        if self.cfg.commands.resample_command_profiles.__contains__("GR1T1-walk"):
            for i in range(len(self.reward_functions_walk)):
                name = self.reward_names_walk[i]
                rew = self.reward_functions_walk[i]() * self.reward_scales_walk[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew

        # walk --------------------------------------------

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    def reset_encoder(self):
        pass

    def post_physics_step_update_state(self):
        super().post_physics_step_update_state()

        if self.cfg.terrain.measure_heights_supervisor:
            self.measured_heights_supervisor = self._get_heights_supervisor()

        self._calculate_feet_orient()
        self._calculate_gait_phase()

    def _calculate_feet_orient(self):
        # feet
        self.left_feet_orient_projected = \
            quat_rotate_inverse(self.rigid_body_states[:, self.feet_indices][:, 0, 3:7], self.gravity_vec)
        self.right_feet_orient_projected = \
            quat_rotate_inverse(self.rigid_body_states[:, self.feet_indices][:, 1, 3:7], self.gravity_vec)
        self.feet_orient_projected = torch.cat((
            self.left_feet_orient_projected.unsqueeze(1),
            self.right_feet_orient_projected.unsqueeze(1)
        ), dim=1)

    def check_termination(self):
        super().check_termination()

        # detect chest tilt too much (roll and pitch)
        if len(self.chest_indices) > 0:
            chest_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.chest_indices][:, 0, 3:7],
                                                          self.gravity_vec)
            self.reset_buf = self.reset_buf | (torch.norm(chest_projected_gravity[:, :2], dim=-1)
                                               > self.cfg.asset.terminate_after_base_projected_gravity_greater_than)

        # detect forehead tilt too much (roll and pitch)
        if len(self.forehead_indices) > 0:
            forehead_projected_gravity = quat_rotate_inverse(
                self.rigid_body_states[:, self.forehead_indices][:, 0, 3:7],
                self.gravity_vec)
            self.reset_buf = self.reset_buf | (torch.norm(forehead_projected_gravity[:, :2], dim=-1)
                                               > self.cfg.asset.terminate_after_base_projected_gravity_greater_than)

    def compute_observation_profile(self):
        if self.obs_profile == "GR1T1-airtime-pri":
            self.obs_buf = torch.cat(
                (
                    # unobservable proprioception
                    # self.base_lin_vel * self.obs_scales.lin_vel * self.lin_vel_scales,
                    # self.base_heights_offset.unsqueeze(1) * self.obs_scales.height_measurements,

                    # imu related
                    # self.imu_ang_vel,
                    # self.imu_projected_gravity,

                    # base related
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    self.base_projected_gravity,
                    self.commands[:, :3] * self.commands_scale,

                    # dof related
                    self.dof_pos_offset * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
                    self.actions,
                ), dim=-1)

            self.pri_obs_buf = torch.cat(
                (
                    # unobservable proprioception
                    self.base_lin_vel * self.obs_scales.lin_vel * self.lin_vel_scales,
                    self.base_heights_offset.unsqueeze(1) * self.obs_scales.height_measurements,

                    # imu related
                    # self.imu_ang_vel,
                    # self.imu_projected_gravity,

                    # base related
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    self.base_projected_gravity,
                    self.commands[:, :3] * self.commands_scale,

                    # dof related
                    self.dof_pos_offset * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
                    self.actions,

                    # height related
                    self.surround_heights_offset_supervisor,

                    # contact
                    self.feet_contact,

                    # foot height
                    self.feet_height,
                ), dim=-1)

    # Jason 2023-11-17
    # 创建 noise vector，此程序只在初始化时调用一次，因此不需要考虑运行效率
    def compute_noise_scale_vec_profile(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])

        if self.obs_profile == "GR1T1-airtime-pri":
            # base related
            noise_vec[0 + 0: 0 + 3] = self.noise_scales.ang_vel * self.noise_level * self.obs_scales.ang_vel
            noise_vec[0 + 3: 3 + 3] = self.noise_scales.gravity * self.noise_level
            noise_vec[3 + 3: 6 + 3] = 0.  # commands (3)

            # dof related
            noise_vec[9 + 0 * self.num_dof: 9 + 1 * self.num_dof] = \
                self.noise_scales.dof_pos * self.noise_level * self.obs_scales.dof_pos
            noise_vec[9 + 1 * self.num_dof: 9 + 2 * self.num_dof] = \
                self.noise_scales.dof_vel * self.noise_level * self.obs_scales.dof_vel
            noise_vec[9 + 2 * self.num_dof: 9 + 3 * self.num_dof] = \
                self.noise_scales.action * self.noise_level * self.obs_scales.action

        # print("noise_vec: ", noise_vec)
        return noise_vec

    def _resample_commands_profile(self,
                                   env_ids,
                                   select_command_profile=None):

        # random swing feet height
        self.swing_feet_height_target[env_ids, 0] = torch_rand_float(self.ranges_swing_feet_height[0],
                                                                     self.ranges_swing_feet_height[1],
                                                                     (len(env_ids), 1),
                                                                     device=self.device).squeeze(1)

        if select_command_profile == "GR1T1-walk":
            # commands
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges_walk["lin_vel_x"][0],
                                                         self.command_ranges_walk["lin_vel_x"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges_walk["lin_vel_y"][0],
                                                         self.command_ranges_walk["lin_vel_y"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)

            # using heading command to auto change the yaw command
            if self.cfg.commands.heading_command:
                self.commands_heading[env_ids] = torch_rand_float(self.command_ranges_walk["heading"][0],
                                                                  self.command_ranges_walk["heading"][1],
                                                                  (len(env_ids), 1),
                                                                  device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges_walk["ang_vel_yaw"][0],
                                                             self.command_ranges_walk["ang_vel_yaw"][1],
                                                             (len(env_ids), 1),
                                                             device=self.device).squeeze(1)

        self._resample_commands_log()

    def _resample_commands_log(self):
        if self.cfg.commands.resample_command_log:
            print("self.commands: \n", self.commands)

    # ----------------------------------------------

    # 惩罚 上半身 Orientation 不水平
    def _reward_cmd_diff_torso_orient(self):
        if len(self.torso_indices) > 0:
            torso_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.torso_indices][:, 0, 3:7],
                                                          self.gravity_vec)
            error_torso_orient = torch.sum(torch.abs(torso_projected_gravity[:, :2]), dim=1)
            reward_torso_orient = torch.exp(self.cfg.rewards.sigma_cmd_diff_torso_orient
                                            * error_torso_orient)
        else:
            reward_torso_orient = 0
        return reward_torso_orient

    def _reward_cmd_diff_chest_orient(self):
        if len(self.chest_indices) > 0:
            chest_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.chest_indices][:, 0, 3:7],
                                                          self.gravity_vec)
            error_chest_orient = torch.sum(torch.abs(chest_projected_gravity[:, :2]), dim=1)
            reward_chest_orient = torch.exp(self.cfg.rewards.sigma_cmd_diff_chest_orient
                                            * error_chest_orient)
        else:
            reward_chest_orient = 0
        return reward_chest_orient

    def _reward_cmd_diff_forehead_orient(self):
        if len(self.forehead_indices) > 0:
            forehead_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.forehead_indices][:, 0, 3:7],
                                                             self.gravity_vec)
            error_forehead_orient = torch.sum(torch.abs(forehead_projected_gravity[:, :2]), dim=1)
            reward_forehead_orient = torch.exp(self.cfg.rewards.sigma_cmd_diff_forehead_orient
                                               * error_forehead_orient)
        else:
            reward_forehead_orient = 0
        return reward_forehead_orient

    # ----------------------------------------------

    def _reward_action_diff_leg(self):
        error_action_diff = (self.last_actions - self.actions) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_leg
                                           * error_action_diff)
        return reward_action_diff

    # 惩罚 髋关节 action 差异
    def _reward_action_diff_hip(self):
        error_action_diff = (self.last_actions[:, self.hip_indices] - self.actions[:, self.hip_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_hip_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_hip
                                               * error_action_diff)
        return reward_hip_action_diff

    def _reward_action_diff_hip_roll(self):
        error_action_diff = (self.last_actions[:, self.hip_roll_indices] - self.actions[:, self.hip_roll_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_hip_roll_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_hip_roll
                                                    * error_action_diff)
        return reward_hip_roll_action_diff

    def _reward_action_diff_hip_yaw(self):
        error_action_diff = (self.last_actions[:, self.hip_yaw_indices] - self.actions[:, self.hip_yaw_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_hip_yaw_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_hip_yaw
                                                   * error_action_diff)
        return reward_hip_yaw_action_diff

    def _reward_action_diff_hip_pitch(self):
        error_action_diff = (self.last_actions[:, self.hip_pitch_indices] - self.actions[:, self.hip_pitch_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_hip_pitch_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_hip_pitch
                                                     * error_action_diff)
        return reward_hip_pitch_action_diff

    # 惩罚 膝关节 action 差异
    def _reward_action_diff_knee(self):
        error_action_diff = (self.last_actions[:, self.knee_indices] - self.actions[:, self.knee_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_knee_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_knee
                                                * error_action_diff)
        return reward_knee_action_diff

    # 惩罚 踝关节 action 差异
    def _reward_action_diff_ankle(self):
        error_action_diff = (self.last_actions[:, self.ankle_indices] - self.actions[:, self.ankle_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_ankle_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_ankle
                                                 * error_action_diff)
        return reward_ankle_action_diff

    def _reward_action_diff_ankle_pitch(self):
        error_action_diff = (self.last_actions[:, self.ankle_pitch_indices] - self.actions[:, self.ankle_pitch_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_ankle_pitch_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_ankle_pitch
                                                       * error_action_diff)
        return reward_ankle_pitch_action_diff

    def _reward_action_diff_ankle_roll(self):
        error_action_diff = (self.last_actions[:, self.ankle_roll_indices] - self.actions[:, self.ankle_roll_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_ankle_roll_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_ankle_roll
                                                      * error_action_diff)
        return reward_ankle_roll_action_diff

    def _reward_action_diff_waist_yaw(self):
        error_action_diff = (self.last_actions[:, self.waist_yaw_indices] - self.actions[:, self.waist_yaw_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_waist_yaw_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_waist_yaw
                                                     * error_action_diff)
        return reward_waist_yaw_action_diff

    def _reward_action_diff_waist_pitch(self):
        error_action_diff = (self.last_actions[:, self.waist_pitch_indices] - self.actions[:, self.waist_pitch_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_waist_pitch_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_waist_pitch
                                                       * error_action_diff)
        return reward_waist_pitch_action_diff

    def _reward_action_diff_waist_roll(self):
        error_action_diff = (self.last_actions[:, self.waist_roll_indices] - self.actions[:, self.waist_roll_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_waist_roll_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_waist_roll
                                                      * error_action_diff)
        return reward_waist_roll_action_diff

    def _reward_action_diff_head(self):
        error_action_diff = (self.last_actions[:, self.head_indices] - self.actions[:, self.head_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_head_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_head
                                                * error_action_diff)
        return reward_head_action_diff

    # 惩罚上半身 action 差异
    def _reward_action_diff_arm(self):
        error_action_diff = (self.last_actions[:, self.arm_indices] - self.actions[:, self.arm_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_arm_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_arm
                                               * error_action_diff)
        return reward_arm_action_diff

    def _reward_action_diff_shoulder_roll(self):
        error_action_diff = (self.last_actions[:, self.shoulder_roll_indices] - self.actions[:, self.shoulder_roll_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_shoulder_roll_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_shoulder_roll
                                                         * error_action_diff)
        return reward_shoulder_roll_action_diff

    def _reward_action_diff_shoulder_yaw(self):
        error_action_diff = (self.last_actions[:, self.shoulder_yaw_indices] - self.actions[:, self.shoulder_yaw_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_shoulder_yaw_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_shoulder_yaw
                                                        * error_action_diff)
        return reward_shoulder_yaw_action_diff

    # ----------------------------------------------

    def _reward_action_diff_diff_hip_roll(self):
        error_action_diff = (self.last_actions[:, self.hip_indices] - self.actions[:, self.hip_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff_last = (self.last_last_actions[:, self.hip_indices] - self.last_actions[:, self.hip_indices]) \
                                 * self.cfg.control.action_scale
        error_action_diff_diff = torch.sum(torch.abs(error_action_diff - error_action_diff_last), dim=1)
        reward_action_diff_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_diff_hip_roll
                                                * error_action_diff_diff)
        return reward_action_diff_diff

    # ----------------------------------------------

    def _reward_dof_vel_new_leg(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel[:, self.leg_indices]), dim=1)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new_leg
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    def _reward_dof_vel_new_knee(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel[:, self.knee_indices]), dim=1)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new_knee
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    def _reward_dof_vel_new_waist_roll(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel[:, self.waist_roll_indices]), dim=1)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new_waist_roll
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    def _reward_dof_vel_new_waist_pitch(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel[:, self.waist_pitch_indices]), dim=1)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new_waist_pitch
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    def _reward_dof_vel_new_head(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel[:, self.head_indices]), dim=1)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new_head
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    def _reward_dof_vel_new_arm(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel[:, self.arm_indices]), dim=1)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new_arm
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    def _reward_dof_vel_new_wrist(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel[:, self.wrist_indices]), dim=1)
        print(error_new_dof_vel)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new_wrist
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    # ----------------------------------------------

    def _reward_dof_acc_new_knee(self):
        error_new_dof_acc = torch.sum(torch.abs((self.last_dof_vel[:, self.knee_indices]
                                                 - self.dof_vel[:, self.knee_indices]) / self.dt), dim=1)
        reward_new_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc_new_knee
                                           * error_new_dof_acc)
        return reward_new_dof_acc

    def _reward_dof_acc_new_waist(self):
        error_new_dof_acc = torch.sum(torch.abs((self.last_dof_vel[:, self.waist_indices]
                                                 - self.dof_vel[:, self.waist_indices]) / self.dt), dim=1)
        reward_new_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc_new_waist
                                           * error_new_dof_acc)
        return reward_new_dof_acc

    def _reward_dof_acc_new_waist_roll(self):
        error_new_dof_acc = torch.sum(torch.abs((self.last_dof_vel[:, self.waist_roll_indices]
                                                 - self.dof_vel[:, self.waist_roll_indices]) / self.dt), dim=1)
        reward_new_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc_new_waist_roll
                                           * error_new_dof_acc)
        return reward_new_dof_acc

    def _reward_dof_acc_new_waist_pitch(self):
        error_new_dof_acc = torch.sum(torch.abs((self.last_dof_vel[:, self.waist_pitch_indices]
                                                 - self.dof_vel[:, self.waist_pitch_indices]) / self.dt), dim=1)
        reward_new_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc_new_waist_pitch
                                           * error_new_dof_acc)
        return reward_new_dof_acc

    def _reward_dof_acc_new_head(self):
        error_new_dof_acc = torch.sum(torch.abs((self.last_dof_vel[:, self.head_indices]
                                                 - self.dof_vel[:, self.head_indices]) / self.dt), dim=1)
        reward_new_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc_new_head
                                           * error_new_dof_acc)
        return reward_new_dof_acc

    def _reward_dof_acc_new_arm(self):
        error_new_dof_acc = torch.sum(torch.abs((self.last_dof_vel[:, self.arm_indices]
                                                 - self.dof_vel[:, self.arm_indices]) / self.dt), dim=1)
        reward_new_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc_new_arm
                                           * error_new_dof_acc)
        return reward_new_dof_acc

    def _reward_dof_acc_new_wrist(self):
        error_new_dof_acc = torch.sum(torch.abs((self.last_dof_vel[:, self.wrist_indices]
                                                 - self.dof_vel[:, self.wrist_indices]) / self.dt), dim=1)
        reward_new_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc_new_wrist
                                           * error_new_dof_acc)
        return reward_new_dof_acc

    # ----------------------------------------------

    def _reward_dof_tor_new_leg(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.leg_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_leg
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_hip(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.hip_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_hip
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_hip_roll(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.hip_roll_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_hip_roll
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_hip_yaw(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.hip_yaw_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_hip_yaw
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_knee(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.knee_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_knee
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_ankle(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.ankle_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_ankle
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_ankle_pitch(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.ankle_pitch_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_ankle_pitch
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_ankle_roll(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.ankle_roll_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_ankle_roll
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_waist(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.waist_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_waist
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_head(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.head_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_head
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_arm(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.arm_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_arm
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_shoulder_yaw(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.shoulder_yaw_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_shoulder_yaw
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_elbow(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.elbow_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_elbow
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_wrist(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.wrist_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_wrist
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_wrist_yaw(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.wrist_yaw_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_wrist_yaw
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    def _reward_dof_tor_new_wrist_pitch(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.wrist_pitch_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_wrist_pitch
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    # ----------------------------------------------
    def _reward_dof_tor_ankle_feet_lift_up(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)

        error_torques_ankle_left_foot_lift_up = torch.sum(
            torch.abs(self.torques[:, self.ankle_indices[:len(self.ankle_indices) // 2]]), dim=1) \
                                                * torch.abs(left_foot_height) \
                                                * (left_foot_height > (self.swing_feet_height_target.squeeze() / 2))
        error_torques_ankle_right_foot_lift_up = torch.sum(
            torch.abs(self.torques[:, self.ankle_indices[len(self.ankle_indices) // 2:]]), dim=1) \
                                                 * torch.abs(right_foot_height) \
                                                 * (right_foot_height > (self.swing_feet_height_target.squeeze() / 2))

        error_dof_tor_ankle_feet_lift_up = error_torques_ankle_left_foot_lift_up + \
                                           error_torques_ankle_right_foot_lift_up

        reward_dof_tor_ankle_feet_lift_up = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_ankle_feet_lift_up
                                                          * error_dof_tor_ankle_feet_lift_up)

        return reward_dof_tor_ankle_feet_lift_up

    # ----------------------------------------------

    # 惩罚 hip roll 的位置偏移
    def _reward_pose_offset_hip_roll(self):
        error_pose_offset_hip_roll = torch.sum(torch.abs(self.dof_pos[:, self.hip_roll_indices]
                                                         - self.default_dof_pos[:, self.hip_roll_indices]), dim=1)

        reward_pose_offset_hip_roll = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_hip_roll
                                                    * error_pose_offset_hip_roll)
        return reward_pose_offset_hip_roll

    # 惩罚 hip yaw 的位置偏移
    def _reward_pose_offset_hip_yaw(self):
        error_pose_offset_hip_yaw = torch.sum(torch.abs(self.dof_pos[:, self.hip_yaw_indices]
                                                        - self.default_dof_pos[:, self.hip_yaw_indices]), dim=1)
        reward_pose_offset_hip_yaw = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_hip_yaw
                                                   * error_pose_offset_hip_yaw)
        return reward_pose_offset_hip_yaw

    # 惩罚 hip roll 的位置偏移 (x 型腿)
    def _reward_pose_offset_hip_roll_x(self):
        error_pose_offset_left_hip_roll_x = self.dof_pos[:, self.hip_roll_indices[0]] \
                                            - self.default_dof_pos[:, self.hip_roll_indices[0]]
        error_pose_offset_left_hip_roll_x *= error_pose_offset_left_hip_roll_x < 0
        error_pose_offset_left_hip_roll_x = torch.abs(error_pose_offset_left_hip_roll_x)

        error_pose_offset_right_hip_roll_x = self.dof_pos[:, self.hip_roll_indices[1]] \
                                             - self.default_dof_pos[:, self.hip_roll_indices[1]]
        error_pose_offset_right_hip_roll_x *= error_pose_offset_right_hip_roll_x > 0
        error_pose_offset_right_hip_roll_x = torch.abs(error_pose_offset_right_hip_roll_x)

        error_pose_offset_hip_roll_x = (error_pose_offset_left_hip_roll_x + error_pose_offset_right_hip_roll_x) \
                                       * (error_pose_offset_left_hip_roll_x < 0) \
                                       * (error_pose_offset_right_hip_roll_x > 0)
        reward_pose_offset_hip_roll_x = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_hip_roll_x
                                                      * error_pose_offset_hip_roll_x)
        return reward_pose_offset_hip_roll_x

    # 惩罚 hip yaw 的位置偏移 (x 型腿)
    def _reward_pose_offset_hip_yaw_x(self):
        error_pose_offset_left_hip_yaw_x = self.dof_pos[:, self.hip_yaw_indices[0]] \
                                           - self.default_dof_pos[:, self.hip_yaw_indices[0]]
        error_pose_offset_left_hip_yaw_x *= error_pose_offset_left_hip_yaw_x < 0
        error_pose_offset_left_hip_yaw_x = torch.abs(error_pose_offset_left_hip_yaw_x)

        error_pose_offset_right_hip_yaw_x = self.dof_pos[:, self.hip_yaw_indices[1]] \
                                            - self.default_dof_pos[:, self.hip_yaw_indices[1]]
        error_pose_offset_right_hip_yaw_x *= error_pose_offset_right_hip_yaw_x > 0
        error_pose_offset_right_hip_yaw_x = torch.abs(error_pose_offset_right_hip_yaw_x)

        error_pose_offset_hip_yaw_x = (error_pose_offset_left_hip_yaw_x + error_pose_offset_right_hip_yaw_x) \
                                      * (error_pose_offset_left_hip_yaw_x < 0) \
                                      * (error_pose_offset_right_hip_yaw_x > 0)
        reward_pose_offset_hip_yaw_x = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_hip_yaw_x
                                                     * error_pose_offset_hip_yaw_x)
        return reward_pose_offset_hip_yaw_x

    # 惩罚 leg 的位置偏移
    def _reward_pose_offset_leg(self):
        error_pose_offset_leg = torch.sum(torch.abs(self.dof_pos[:, self.leg_indices]
                                                    - self.default_dof_pos[:, self.leg_indices]), dim=1)
        reward_pose_offset_leg = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_leg
                                               * error_pose_offset_leg)
        return reward_pose_offset_leg

    def _reward_pose_offset_knee(self):
        error_pose_offset_knee = torch.sum(torch.abs(self.dof_pos[:, self.knee_indices]
                                                     - self.default_dof_pos[:, self.knee_indices]), dim=1)
        reward_pose_offset_knee = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_knee
                                                * error_pose_offset_knee)
        return reward_pose_offset_knee

    def _reward_pose_offset_ankle(self):
        error_pose_offset_ankle = torch.sum(torch.abs(self.dof_pos[:, self.ankle_indices]
                                                      - self.default_dof_pos[:, self.ankle_indices]), dim=1)
        reward_pose_offset_ankle = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_ankle
                                                 * error_pose_offset_ankle)
        return reward_pose_offset_ankle

    def _reward_pose_offset_ankle_roll(self):
        error_pose_offset_ankle_roll = torch.sum(torch.abs(self.dof_pos[:, self.ankle_roll_indices]
                                                           - self.default_dof_pos[:, self.ankle_roll_indices]), dim=1)
        reward_pose_offset_ankle_roll = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_ankle_roll
                                                      * error_pose_offset_ankle_roll)
        return reward_pose_offset_ankle_roll

    def _reward_pose_offset_waist(self):
        error_pose_offset_waist = torch.sum(torch.abs(self.dof_pos[:, self.waist_indices]
                                                      - self.default_dof_pos[:, self.waist_indices]), dim=1)
        reward_pose_offset_waist = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_waist
                                                 * error_pose_offset_waist)
        return reward_pose_offset_waist

    def _reward_pose_offset_waist_yaw(self):
        error_pose_offset_waist_yaw = torch.sum(torch.abs(self.dof_pos[:, self.waist_yaw_indices]
                                                          - self.default_dof_pos[:, self.waist_yaw_indices]),
                                                dim=1)
        reward_pose_offset_waist_yaw = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_waist_yaw
                                                     * error_pose_offset_waist_yaw)
        return reward_pose_offset_waist_yaw

    def _reward_pose_offset_waist_roll(self):
        error_pose_offset_waist_roll = torch.sum(torch.abs(self.dof_pos[:, self.waist_roll_indices]
                                                           - self.default_dof_pos[:, self.waist_roll_indices]),
                                                 dim=1)
        reward_pose_offset_waist_roll = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_waist_roll
                                                      * error_pose_offset_waist_roll)
        return reward_pose_offset_waist_roll

    def _reward_pose_offset_waist_pitch(self):
        error_pose_offset_waist_pitch = torch.sum(torch.abs(self.dof_pos[:, self.waist_pitch_indices]
                                                            - self.default_dof_pos[:, self.waist_pitch_indices]),
                                                  dim=1)
        reward_pose_offset_waist_pitch = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_waist_pitch
                                                       * error_pose_offset_waist_pitch)
        return reward_pose_offset_waist_pitch

    def _reward_pose_offset_head(self):
        error_pose_offset_head = torch.sum(torch.abs(self.dof_pos[:, self.head_indices]
                                                     - self.default_dof_pos[:, self.head_indices]), dim=1)
        reward_pose_offset_head = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_head
                                                * error_pose_offset_head)
        return reward_pose_offset_head

    def _reward_pose_offset_head_roll(self):
        error_pose_offset_head_roll = torch.sum(torch.abs(self.dof_pos[:, self.head_roll_indices]
                                                          - self.default_dof_pos[:, self.head_roll_indices]),
                                                dim=1)
        reward_pose_offset_head_roll = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_head_roll
                                                     * error_pose_offset_head_roll)
        return reward_pose_offset_head_roll

    def _reward_pose_offset_head_pitch(self):
        error_pose_offset_head_pitch = torch.sum(torch.abs(self.dof_pos[:, self.head_pitch_indices]
                                                           - self.default_dof_pos[:, self.head_pitch_indices]),
                                                 dim=1)
        reward_pose_offset_head_pitch = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_head_pitch
                                                      * error_pose_offset_head_pitch)
        return reward_pose_offset_head_pitch

    # 惩罚 arm 的位置偏移
    def _reward_pose_offset_arm(self):
        error_pose_offset_arm = torch.sum(torch.abs(self.dof_pos[:, self.arm_indices]
                                                    - self.default_dof_pos[:, self.arm_indices]), dim=1)
        reward_pose_offset_arm = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_arm
                                               * error_pose_offset_arm)
        return reward_pose_offset_arm

    def _reward_pose_offset_shoulder_pitch(self):
        error_pose_offset_shoulder_pitch = torch.sum(torch.abs(self.dof_pos[:, self.shoulder_pitch_indices]
                                                               - self.default_dof_pos[:, self.shoulder_pitch_indices]),
                                                     dim=1)
        reward_pose_offset_shoulder_pitch = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_shoulder_pitch
                                                          * error_pose_offset_shoulder_pitch)
        return reward_pose_offset_shoulder_pitch

    def _reward_pose_offset_shoulder_roll(self):
        error_pose_offset_shoulder_roll = torch.sum(torch.abs(self.dof_pos[:, self.shoulder_roll_indices]
                                                              - self.default_dof_pos[:, self.shoulder_roll_indices]),
                                                    dim=1)
        reward_pose_offset_shoulder_roll = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_shoulder_roll
                                                         * error_pose_offset_shoulder_roll)
        return reward_pose_offset_shoulder_roll

    def _reward_pose_offset_shoulder_yaw(self):
        error_pose_offset_shoulder_yaw = torch.sum(torch.abs(self.dof_pos[:, self.shoulder_yaw_indices]
                                                             - self.default_dof_pos[:, self.shoulder_yaw_indices]),
                                                   dim=1)
        reward_pose_offset_shoulder_yaw = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_shoulder_yaw
                                                        * error_pose_offset_shoulder_yaw)
        return reward_pose_offset_shoulder_yaw

    def _reward_pose_offset_elbow(self):
        error_pose_offset_elbow = torch.sum(torch.abs(self.dof_pos[:, self.elbow_indices]
                                                      - self.default_dof_pos[:, self.elbow_indices]), dim=1)
        reward_pose_offset_elbow = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_elbow
                                                 * error_pose_offset_elbow)
        return reward_pose_offset_elbow

    def _reward_pose_offset_wrist(self):
        error_pose_offset_wrist = torch.sum(torch.abs(self.dof_pos[:, self.wrist_indices]
                                                      - self.default_dof_pos[:, self.wrist_indices]), dim=1)
        reward_pose_offset_wrist = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_wrist
                                                 * error_pose_offset_wrist)
        return reward_pose_offset_wrist

    def _reward_pose_offset_wrist_yaw(self):
        error_pose_offset_wrist_yaw = torch.sum(torch.abs(self.dof_pos[:, self.wrist_yaw_indices]
                                                          - self.default_dof_pos[:, self.wrist_yaw_indices]), dim=1)
        reward_pose_offset_wrist_yaw = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_wrist_yaw
                                                     * error_pose_offset_wrist_yaw)
        return reward_pose_offset_wrist_yaw

    def _reward_pose_offset_wrist_roll(self):
        error_pose_offset_wrist_roll = torch.sum(torch.abs(self.dof_pos[:, self.wrist_roll_indices]
                                                           - self.default_dof_pos[:, self.wrist_roll_indices]), dim=1)
        reward_pose_offset_wrist_roll = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_wrist_roll
                                                      * error_pose_offset_wrist_roll)
        return reward_pose_offset_wrist_roll

    def _reward_pose_offset_wrist_pitch(self):
        error_pose_offset_wrist_pitch = torch.sum(torch.abs(self.dof_pos[:, self.wrist_pitch_indices]
                                                            - self.default_dof_pos[:, self.wrist_pitch_indices]), dim=1)
        reward_pose_offset_wrist_pitch = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_wrist_pitch
                                                       * error_pose_offset_wrist_pitch)
        return reward_pose_offset_wrist_pitch

    # ----------------------------------------------

    # 惩罚 上半身 Ang Vel
    def _reward_pose_vel_waist(self):
        error_ang_vel_waist = torch.sum(torch.abs(self.rigid_body_states[:, self.waist_indices][:, 0, 10:12]), dim=-1)
        reward_ang_vel_waist = 1 - torch.exp(self.cfg.rewards.sigma_pose_vel_waist
                                             * error_ang_vel_waist)
        return reward_ang_vel_waist

    def _reward_pose_vel_head(self):
        error_ang_vel_head = torch.sum(torch.abs(self.rigid_body_states[:, self.head_indices][:, 0, 10:12]), dim=-1)
        reward_ang_vel_head = 1 - torch.exp(self.cfg.rewards.sigma_pose_vel_head
                                            * error_ang_vel_head)
        return reward_ang_vel_head

    # ----------------------------------------------

    # 惩罚 脚部 接近地面时不水平
    def _reward_orient_diff_feet_put_down(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)

        # Jason 2023-12-27:
        # normalize the error by the target height
        error_distance_to_ground_left_foot = torch.abs(left_foot_height - self.swing_feet_height_target) \
                                             / self.swing_feet_height_target.squeeze()
        error_distance_to_ground_right_foot = torch.abs(right_foot_height - self.swing_feet_height_target) \
                                              / self.swing_feet_height_target.squeeze()

        error_orient_diff_left_foot = torch.sum(torch.abs(self.left_feet_orient_projected[:, :2]), dim=1) \
                                      * (error_distance_to_ground_left_foot ** 2)
        error_orient_diff_right_foot = torch.sum(torch.abs(self.right_feet_orient_projected[:, :2]), dim=1) \
                                       * (error_distance_to_ground_right_foot ** 2)

        error_orient_diff_feet_put_down = error_orient_diff_left_foot + error_orient_diff_right_foot
        reward_orient_diff_feet_put_down = 1 - torch.exp(self.cfg.rewards.sigma_orient_diff_feet_put_down
                                                         * error_orient_diff_feet_put_down)
        return reward_orient_diff_feet_put_down

    def _reward_orient_diff_feet_lift_up(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)

        # Jason 2023-12-27:
        # normalize the error by the target height
        error_distance_to_height_target_left_foot = torch.abs(left_foot_height * (left_foot_height > 0)) \
                                                    / self.swing_feet_height_target.squeeze()
        error_distance_to_height_target_right_foot = torch.abs(right_foot_height * (right_foot_height > 0)) \
                                                     / self.swing_feet_height_target.squeeze()

        error_orient_diff_left_foot = torch.sum(torch.abs(self.left_feet_orient_projected[:, :2]), dim=1) \
                                      * (error_distance_to_height_target_left_foot ** 2)
        error_orient_diff_right_foot = torch.sum(torch.abs(self.right_feet_orient_projected[:, :2]), dim=1) \
                                       * (error_distance_to_height_target_right_foot ** 2)

        error_orient_diff_feet_lift_up = error_orient_diff_left_foot + error_orient_diff_right_foot
        reward_orient_diff_feet_lift_up = 1 - torch.exp(self.cfg.rewards.sigma_orient_diff_feet_lift_up
                                                        * error_orient_diff_feet_lift_up)
        return reward_orient_diff_feet_lift_up

    # ----------------------------------------------

    def _reward_feet_speed_xy_close_to_ground(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)

        error_left_foot_speed_xy_close_to_ground = \
            torch.norm(self.avg_feet_speed_xyz[:, 0, :2], dim=1) \
            * torch.abs(left_foot_height - self.swing_feet_height_target.squeeze() / 2) \
            * (left_foot_height < self.swing_feet_height_target.squeeze() / 2)
        error_right_foot_speed_xy_close_to_ground = \
            torch.norm(self.avg_feet_speed_xyz[:, 1, :2], dim=1) \
            * torch.abs(right_foot_height - self.swing_feet_height_target.squeeze() / 2) \
            * (right_foot_height < self.swing_feet_height_target.squeeze() / 2)

        error_feet_speed_xy_close_to_ground = error_left_foot_speed_xy_close_to_ground + \
                                              error_right_foot_speed_xy_close_to_ground

        reward_feet_speed_xy_close_to_ground = 1 - torch.exp(self.cfg.rewards.sigma_feet_speed_xy_close_to_ground
                                                             * error_feet_speed_xy_close_to_ground)
        return reward_feet_speed_xy_close_to_ground

    def _reward_feet_speed_z_close_to_height_target(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1)
            - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1)
            - self.measured_heights_supervisor, dim=1)

        error_left_foot_speed_z_close_to_height_target = \
            torch.abs(self.avg_feet_speed_xyz[:, 0, 2]) \
            * torch.abs(left_foot_height - self.swing_feet_height_target.squeeze() / 2) \
            * (left_foot_height > self.swing_feet_height_target.squeeze() / 2)
        error_right_foot_speed_z_close_to_height_target = \
            torch.abs(self.avg_feet_speed_xyz[:, 1, 2]) \
            * torch.abs(right_foot_height - self.swing_feet_height_target.squeeze() / 2) \
            * (right_foot_height > self.swing_feet_height_target.squeeze() / 2)

        error_feet_speed_z_close_to_height_target = error_left_foot_speed_z_close_to_height_target + \
                                                    error_right_foot_speed_z_close_to_height_target

        reward_feet_speed_z_close_to_height_target = 1 - torch.exp(
            self.cfg.rewards.sigma_feet_speed_z_close_to_height_target
            * error_feet_speed_z_close_to_height_target)

        return reward_feet_speed_z_close_to_height_target

    # ----------------------------------------------

    def _reward_feet_air_time(self):
        # 计算 first_contact 的个数，如果有接触到地面，则将 feet_air_time 置为 0
        feet_air_time_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target
        feet_air_time_error = torch.abs(feet_air_time_error)

        reward_feet_air_time = torch.exp(self.cfg.rewards.sigma_feet_air_time
                                         * feet_air_time_error)
        reward_feet_air_time *= self.feet_first_contact
        reward_feet_air_time = torch.sum(reward_feet_air_time, dim=1)
        reward_feet_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.05  # no reward for zero command

        return reward_feet_air_time

    def _reward_feet_air_height(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1)
            - self.measured_heights_supervisor,
            dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1)
            - self.measured_heights_supervisor,
            dim=1)

        stack_feet_height = torch.stack((left_foot_height,
                                         right_foot_height))
        min_feet_height, min_feet_height_index = torch.min(stack_feet_height, dim=0)

        error_feet_air_height_left_foot = torch.abs(left_foot_height
                                                    - min_feet_height
                                                    - self.swing_feet_height_target.squeeze())
        error_feet_air_height_right_foot = torch.abs(right_foot_height
                                                     - min_feet_height
                                                     - self.swing_feet_height_target.squeeze())

        # Jason 2023-12-25:
        # 用二次项来描述，更加关注于指定时间段内的高度差
        reward_feet_air_height_left_foot = torch.exp(self.cfg.rewards.sigma_feet_air_height
                                                     * error_feet_air_height_left_foot)
        reward_feet_air_height_right_foot = torch.exp(self.cfg.rewards.sigma_feet_air_height
                                                      * error_feet_air_height_right_foot)

        reward_feet_air_height = torch.stack((reward_feet_air_height_left_foot,
                                              reward_feet_air_height_right_foot), dim=1)

        # Jason 2024-03-31:
        # use air time to catch period at height target
        feet_air_time_mid_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target / 2
        feet_air_time_mid_error = torch.abs(feet_air_time_mid_error)
        feet_air_time_mid_error = torch.exp(self.cfg.rewards.sigma_feet_air_time_mid
                                            * feet_air_time_mid_error)

        reward_feet_air_height = feet_air_time_mid_error * reward_feet_air_height
        reward_feet_air_height = torch.sum(reward_feet_air_height, dim=1)
        reward_feet_air_height *= torch.norm(self.commands[:, :2], dim=1) > 0.05  # no reward for zero command

        return reward_feet_air_height

    def _reward_feet_air_force(self):
        reward_feet_air_force_left_foot = torch.exp(self.cfg.rewards.sigma_feet_force
                                                    * torch.abs(self.avg_feet_contact_force[:, 0]))
        reward_feet_air_force_right_foot = torch.exp(self.cfg.rewards.sigma_feet_force
                                                     * torch.abs(self.avg_feet_contact_force[:, 1]))

        reward_feet_air_force = torch.stack((reward_feet_air_force_left_foot,
                                             reward_feet_air_force_right_foot), dim=1)

        # Jason 2024-03-31:
        # use air time to catch period at height target
        feet_air_time_mid_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target / 2
        feet_air_time_mid_error = torch.abs(feet_air_time_mid_error)
        feet_air_time_mid_error = torch.exp(self.cfg.rewards.sigma_feet_air_time_mid
                                            * feet_air_time_mid_error)

        reward_feet_air_force = feet_air_time_mid_error * reward_feet_air_force
        reward_feet_air_force = torch.sum(reward_feet_air_force, dim=1)
        reward_feet_air_force *= torch.norm(self.commands[:, :2], dim=1) > 0.05  # no reward for zero command

        return reward_feet_air_force

    def _reward_feet_land_time(self):
        # 计算 first_contact 的个数，如果有接触到地面，则将 feet_land_time 置为 0
        feet_land_time_error = (self.feet_land_time - self.cfg.rewards.feet_land_time_max) \
                               * (self.feet_land_time > self.cfg.rewards.feet_land_time_max)

        reward_feet_land_time = 1 - torch.exp(self.cfg.rewards.sigma_feet_land_time
                                              * feet_land_time_error)
        reward_feet_land_time = torch.sum(reward_feet_land_time, dim=1)
        reward_feet_land_time *= torch.norm(self.commands[:, :2], dim=1) > 0.05  # no reward for zero command

        return reward_feet_land_time

    def _reward_on_the_air(self):
        # 惩罚两条腿都没有接触到地面的情况
        jumping_error = torch.sum(self.feet_contact, dim=1) == 0

        # use exponential to make the reward more sparse
        reward_jumping = jumping_error
        return reward_jumping

    # ----------------------------------------------

    def _reward_hip_yaw(self):
        # print(torch.sum(torch.abs(self.dof_pos[:,[1,7]]- self.default_dof_pos[:,[1,7]]),dim=1))
        # print(torch.abs(self.commands[:, 2]))
        return torch.sum(torch.abs(self.dof_pos[:, [1, 7]]), dim=1)

    def _reward_hip_roll(self):
        # print(torch.sum(torch.abs(self.dof_pos[:,[1,7]]- self.default_dof_pos[:,[1,7]]),dim=1))
        # print(torch.abs(self.commands[:, 2]))
        return torch.sum(torch.abs(self.dof_vel[:, [0, 6]]), dim=1)

    # ----------------------------------------------

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        left_foot_fxy = torch.norm(self.contact_forces[:, self.feet_indices][:, 0, :2], dim=1)
        right_foot_fxy = torch.norm(self.contact_forces[:, self.feet_indices][:, 1, :2], dim=1)

        left_foot_fz = self.contact_forces[:, self.feet_indices][:, 0, 2]
        right_foot_fz = self.contact_forces[:, self.feet_indices][:, 1, 2]

        error_left_foot_f = left_foot_fxy - self.cfg.rewards.feet_stumble_ratio * torch.abs(left_foot_fz)
        error_right_foot_f = right_foot_fxy - self.cfg.rewards.feet_stumble_ratio * torch.abs(right_foot_fz)

        error_left_foot_f = error_left_foot_f * (error_left_foot_f > 0)
        error_right_foot_f = error_right_foot_f * (error_right_foot_f > 0)

        # print("error_left_foot_f = \n", error_left_foot_f)

        reward_left_foot_f = 1 - torch.exp(self.cfg.rewards.sigma_feet_stumble * error_left_foot_f)
        reward_right_foot_f = 1 - torch.exp(self.cfg.rewards.sigma_feet_stumble * error_right_foot_f)

        reward_feet_stumble = reward_left_foot_f + reward_right_foot_f
        return reward_feet_stumble