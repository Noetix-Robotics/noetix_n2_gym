from humanoid.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
 
LEFT_YAW_ROLL  = [0, 1]        
RIGHT_YAW_ROLL  = [5, 6]   

class N2Env(LeggedRobot):

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0. # commands
        noise_vec[3:5] = 0. # sin/cos phase
        noise_vec[5:8] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[8:10] = noise_scales.quat * noise_level
        noise_vec[10:10+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[10+self.num_actions:10+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[10+2*self.num_actions:10+3*self.num_actions] = 0. # previous actions

        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = self.cfg.rewards.cycle_time
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if len(env_ids) > 0:
            random_tensor = torch.rand_like(self.commands[env_ids, 0])
            self.commands[env_ids[random_tensor < 0.20], :] = 0
            self.commands[env_ids[torch.logical_and(random_tensor >= 0.20, random_tensor < 0.30)], 0] = 0
            self.commands[env_ids[torch.logical_and(random_tensor >= 0.30, random_tensor < 0.40)], 2] = 0

        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > self.min_cmd_vel).unsqueeze(1)
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  
                                    self.commands[:, :3] * self.commands_scale,
                                    sin_phase,
                                    cos_phase,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.base_euler_xyz[:, :2],  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                ),dim=-1)
        self.privileged_obs_buf = torch.cat((  
                                    self.commands[:, :3] * self.commands_scale,
                                    sin_phase,
                                    cos_phase,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.base_euler_xyz[:, :2],  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel
                                ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

# ================================================ Rewards ================================================== #
    def _reward_contact(self):
        # 初始化奖励张量
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        for i in range(self.feet_num):
            # 获取相位和接触状态
            phase = self.leg_phase[:, i]
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            
            # --- 定义阶段状态 ---
            is_stance = phase < 0.55        # 支撑阶段（相位 < 55%）
            is_swing = ~is_stance           # 摆动阶段（相位 >= 55%）
            
            # --- 奖励逻辑 ---
            # 正确行为：支撑阶段触地 → +1奖励
            correct_contact = is_stance & contact
            res += correct_contact.float()
            
            # --- 惩罚逻辑 ---
            # 错误行为：摆动阶段触地 → 拖步惩罚
            swing_dragging = is_swing & contact
            swing_penalty = swing_dragging.float() * contact * 0.5  # 惩罚系数0.5
            
            # 合并惩罚到总奖励
            res += (-swing_penalty)  # 惩罚为负值
        
        return res
    
    def _reward_feet_contact(self):
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 5.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        single_feet_contact = torch.logical_xor(self.contact_filt[:, 0], self.contact_filt[:, 1]) 
        both_feet_contact = torch.logical_and(self.contact_filt[:, 0], self.contact_filt[:, 1])

        running_envs = self.commands[:, 0] > 1.0
        walking_envs = ~running_envs

        self.feet_contact_time[both_feet_contact] += self.dt
        self.feet_contact_time *= both_feet_contact
        
        walking_rew_filter = torch.logical_or(single_feet_contact, self.feet_contact_time < 0.2)
        rew[walking_envs & walking_rew_filter] = 1.0

        rew[self.zero_cmd_ids] = 1
        return rew
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 5.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.025 - self.cfg.rewards.target_feet_height) * ~contact
        return torch.sum(pos_error, dim=(1,))
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 5.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_dof_pos

        left_yaw_roll = joint_diff[:, LEFT_YAW_ROLL]
        right_yaw_roll = joint_diff[:, RIGHT_YAW_ROLL]

        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

        rew = torch.exp(-yaw_roll * 100) - 0.05 * torch.norm(joint_diff[:, LEFT_YAW_ROLL + RIGHT_YAW_ROLL], dim=1)
        rew[self.commands[:, 1] > self.min_cmd_vel] = 1.   
        rew[self.commands[:, 2] > self.min_cmd_vel] = 1.     
        return rew
    