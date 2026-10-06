import argparse
import json
import math
import random
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Compare PPO checkpoints in a fixed-gravity simulation.")
parser.add_argument("--task", default="Isaac-Inhand-Rotate-Sharpa-Wave-v0")
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--robust", action="store_true", help="Evaluate with training-time dynamics randomization.")
parser.add_argument("--randomize_pd_gains", action="store_true")
parser.add_argument("--randomize_com", action="store_true")
parser.add_argument("--randomize_mass", action="store_true")
parser.add_argument("--output", default="logs/checkpoint_comparison.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
from isaaclab_tasks.utils.hydra import hydra_task_config

from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

import rl_isaaclab.tasks.inhand_rotate


def evaluate_checkpoint(agent, env, checkpoint):
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    agent.restore_test(checkpoint)
    agent.set_eval()
    observations = env.reset()
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    episode_net_rotation = torch.zeros(env.num_envs, device=env.device)
    completed_rewards = []
    completed_lengths = []
    completed_net_rotations = []
    height_terminations = 0
    timeouts = 0
    full_rotation_and_hold = 0
    reward_sum = torch.zeros((), device=env.device)
    rotation_sum = torch.zeros((), device=env.device)
    yaw_sum = torch.zeros((), device=env.device)
    position_bonus_sum = torch.zeros((), device=env.device)
    rotation_bonus_sum = torch.zeros((), device=env.device)
    integrated_rotation_sum = torch.zeros((), device=env.device)

    with torch.no_grad():
        for _ in range(args_cli.steps):
            policy_input = {
                "obs": agent.running_mean_std(observations["obs"]),
                "priv_info": observations["priv_info"],
            }
            actions = torch.clamp(agent.model.act_inference(policy_input), -1.0, 1.0)
            observations, rewards, dones, info = env.step(actions)
            rotation_delta = axis_angle_from_quat(
                quat_mul(env.unwrapped.object_rot, quat_conjugate(env.unwrapped.object_rot_prev))
            )[:, 2]
            episode_rewards += rewards
            episode_lengths += 1
            episode_net_rotation += rotation_delta
            reward_sum += rewards.sum()
            rotation_sum += info["rotate_reward"]
            yaw_sum += info["yaw"]
            position_bonus_sum += info["object_pos_diff"] * env.cfg.object_pos_reward_scale
            rotation_bonus_sum += info["rotate_reward"] * env.cfg.rotate_reward_scale
            integrated_rotation_sum += rotation_delta.sum()

            finished = dones.bool()
            if finished.any():
                completed_rewards.extend(episode_rewards[finished].cpu().tolist())
                completed_lengths.extend(episode_lengths[finished].cpu().tolist())
                completed_net_rotations.extend(episode_net_rotation[finished].cpu().tolist())
                finished_timeouts = info["time_outs"][finished].bool()
                timeouts += int(finished_timeouts.sum().item())
                height_terminations += int((~finished_timeouts).sum().item())
                full_rotation_and_hold += int(
                    ((episode_net_rotation[finished] >= 2 * math.pi) & finished_timeouts).sum().item()
                )
                episode_rewards[finished] = 0
                episode_lengths[finished] = 0
                episode_net_rotation[finished] = 0

    completed = len(completed_rewards)
    return {
        "checkpoint": checkpoint,
        "steps": args_cli.steps,
        "num_envs": env.num_envs,
        "seed": args_cli.seed,
        "gravity_z_m_s2": -9.81,
        "dynamics_randomization": {
            "pd_gains": env.cfg.randomize_pd_gains,
            "center_of_mass": env.cfg.randomize_com,
            "mass": env.cfg.randomize_mass,
        },
        "completed_episodes": completed,
        "height_terminations": height_terminations,
        "timeouts": timeouts,
        "timeout_fraction": timeouts / completed if completed else None,
        "full_rotation_and_hold": full_rotation_and_hold,
        "full_rotation_and_hold_fraction": full_rotation_and_hold / completed if completed else None,
        "mean_net_rotations_per_episode": sum(completed_net_rotations) / (2 * math.pi * completed) if completed else None,
        "mean_episode_reward": sum(completed_rewards) / completed if completed else None,
        "mean_episode_steps": sum(completed_lengths) / completed if completed else None,
        "mean_step_reward": (reward_sum / (args_cli.steps * env.num_envs)).item(),
        "mean_rewarded_target_angular_velocity_rad_s": (rotation_sum / args_cli.steps).item(),
        "mean_yaw_velocity_rad_s": (yaw_sum / args_cli.steps).item(),
        "mean_integrated_yaw_velocity_rad_s": (
            integrated_rotation_sum / (args_cli.steps * env.num_envs * env.unwrapped.step_dt)
        ).item(),
        "mean_position_reward_per_step": (position_bonus_sum / args_cli.steps).item(),
        "mean_rotation_reward_per_step": (rotation_bonus_sum / args_cli.steps).item(),
    }


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.gravity_curriculum = False
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = args_cli.robust or args_cli.randomize_pd_gains
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = args_cli.robust or args_cli.randomize_com
    env_cfg.randomize_mass = args_cli.robust or args_cli.randomize_mass
    env_cfg.randomize_joint_pos_offset = args_cli.robust
    agent_cfg["device"] = args_cli.device
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["algorithm"]["num_actors"] = args_cli.num_envs
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    try:
        agent = PPO(env, output_dir="", full_config=config, create_output_dir=False)
        results = []
        for checkpoint in args_cli.checkpoints:
            print(f"Evaluating {checkpoint}", flush=True)
            result = evaluate_checkpoint(agent, env, checkpoint)
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)

        with open(args_cli.output, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)
        print(f"Saved comparison to {args_cli.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
