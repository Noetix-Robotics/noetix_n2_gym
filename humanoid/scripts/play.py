import os
import sys
from humanoid import LEGGED_GYM_ROOT_DIR

import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = 'plane' # plane trimesh
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    if CONTROL_ROBOT:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        env_cfg.env.episode_length_s = 100
        env_cfg.commands.resampling_time = 1000
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.policy, path, ppo_runner.obs_normalizer)
        export_policy_as_onnx(ppo_runner.alg.policy, path, ppo_runner.obs_normalizer)
        print('Exported policy to: ', path)

    for i in range(10*int(env.max_episode_length)):
        env.update_keyboard_events()
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    CONTROL_ROBOT = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
