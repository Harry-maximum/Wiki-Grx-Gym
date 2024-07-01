# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.gr1t1.gr1t1_lower_limb_config import GR1T1LowerLimbCfg ##lower limb cfg version
import torch


class cmd:
    vx = 0.3
    vy = 0
    dyaw = 0

def quat_rotate_inverse(quat, gvec):

    quat = quat.reshape(4,1)
    gvec = gvec.reshape(3,1)
    
    quat_w = quat[3, 0]
    quat_vec = quat[:3, 0]  # Vector part

    a = gvec * (2.0 * quat_w ** 2 - 1.0)
    b = 2.0 * quat_w * np.cross(quat_vec, gvec.flatten()[:, np.newaxis], axis=0)
    c = 2.0 * quat_vec[:, np.newaxis] * np.dot(quat_vec.T, gvec)
    
    # Compute the rotated vector
    result = a - b + c
    
    return result.flatten()

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1,2,3,0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec,r)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    print(f"hist_obs: {len(hist_obs)}")
    count_lowlevel = 0


    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec , r= get_obs(data)

        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        default_joint_angles = np.array([cfg.init_state.default_joint_angles[key] for key in sorted(cfg.init_state.default_joint_angles.keys())])
        default_joint_angles = default_joint_angles[-cfg.env.num_actions:]

        # 1000hz -> 50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            quat_proj = quat_rotate_inverse(quat, gvec)
            print(f"omega shape: {omega.shape}")
            print(f"omega: {omega}")
            print(f"eu_ang shape: {eu_ang.shape}")
            print(f"obs shape: {obs.shape}")
            #obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            #obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 0:3] = omega / 3.14 * 180
            obs[0, 3:6] = quat_proj
            obs[0, 6] = cmd.vx 
            obs[0, 7] = cmd.vy 
            obs[0, 8] = cmd.dyaw
            obs[0, 9:19] = (q - default_joint_angles) * cfg.normalization.obs_scales.dof_pos 
            obs[0, 19:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:39] = action
            
            print(f"obs shape: {obs.shape}")
            print(f"obs: {obs}")
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            print(f"obs shape: {obs.shape}")
            hist_obs.append(obs)
            hist_obs.popleft()
            print(f"hist_obs 长度: {len(hist_obs)}")
            for i, hist in enumerate(hist_obs):
                print(f"hist_obs[{i}] 形状: {hist.shape}")

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            
            policy_input[0, :cfg.env.num_single_obs] = obs[0, :cfg.env.num_single_obs]   
                     
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, cfg.normalization.clip_actions_min, cfg.normalization.clip_actions_max)
           
            target_q = (action + default_joint_angles) * cfg.control.action_scale
            target_dq = action / (model.opt.timestep * cfg.sim_config.decimation)
            print(f"degault_joint_angle{default_joint_angles}")

        # Generate PD control
        tau = (target_q-q) * cfg.robot_config.kps - dq * cfg.robot_config.kds
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()
    
    class Sim2simCfg(GR1T1LowerLimbCfg):

        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/fourier_intelligence_gr1t1/scene.xml'
           
            sim_duration = 70.0
            dt = 0.001
            decimation = 20

        class robot_config:
            
            kps = np.array([57, 43, 114, 114, 15.3, 
                            57, 43, 114, 114, 15.3], dtype=np.double)  ##114
            kds = np.array([5.7, 4.3, 11.4, 11.4, 1.53, 
                            5.7, 4.3, 11.4, 11.4, 1.53], dtype=np.double)
            tau_limit = np.array([60, 45, 130, 130, 16, 
                                60, 45, 130, 130, 16], dtype=np.double)
           
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
