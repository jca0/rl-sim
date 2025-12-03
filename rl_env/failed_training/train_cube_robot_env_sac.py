#!/usr/bin/env python3
"""SAC trainer for RedCubePick using Stable Baselines3 with video/joint logging."""

import argparse
import json
import time
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import rl_env  # registers the env

ENV_ID = "RedCubePick-v0"
VIDEO_FPS = 30
EVAL_MAX_STEPS = 600

# Fixed SAC hyperparameters
TOTAL_TIMESTEPS = 1_000_000
VIDEO_INTERVAL = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train RedCubePick SAC with logging artifacts.")
    parser.add_argument("--exp-name", type=str, default="red_cube_pick_sac")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--video-interval", type=int, default=VIDEO_INTERVAL,
                        help="Steps between saved rollout videos (0 disables).")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--save-policy", action="store_true", help="Persist final weights.")
    return parser.parse_args()


def make_env(seed: int, idx: int) -> gym.Env:
    def thunk() -> gym.Env:
        env = gym.make(ENV_ID)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # ClipAction sets action space to (-inf, inf), which SAC rejects.
        # env = gym.wrappers.ClipAction(env)
        # Note: Using VecNormalize in main() instead of wrappers.NormalizeObservation here
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk


class VideoAndJointLogCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: Path, seed: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.seed = seed
        self.video_dir = log_dir / "videos"
        self.joint_dir = log_dir / "joint_logs"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.joint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.record_rollout()
        return True

    def record_rollout(self):
        env = gym.make(ENV_ID, render_mode="rgb_array")
        obs, _ = env.reset(seed=self.seed)
        
        frames: List[np.ndarray] = []
        true_env = env.unwrapped
        joints_log: List[Dict[str, Any]] = [
            {"time_s": 0.0, "qpos": true_env.current_joint_positions().tolist()}
        ]
        elapsed = 0.0
        next_sample = 1.0
        total_reward = 0.0
        
        # Handle VecNormalize if present in the model
        vec_norm = self.model.get_vec_normalize_env()
        
        for _ in range(EVAL_MAX_STEPS):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Prepare observation for prediction
            if vec_norm is not None:
                # Normalize the observation using the training env statistics
                # VecNormalize expects (n_envs, obs_dim)
                obs_tensor = obs[np.newaxis, ...]
                obs_norm = vec_norm.normalize_obs(obs_tensor)
                action, _ = self.model.predict(obs_norm, deterministic=True)
                action = action[0] # Unpack from batch
            else:
                action, _ = self.model.predict(obs, deterministic=True)
            
            # Env step
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            elapsed += true_env.dt
            
            while elapsed >= next_sample:
                joints_log.append(
                    {
                        "time_s": round(float(next_sample), 4),
                        "qpos": true_env.current_joint_positions().tolist(),
                    }
                )
                next_sample += 1.0
            
            if terminated or truncated:
                break
        
        env.close()
        
        if not frames:
            print("No frames captured!")
            return

        global_step = self.num_timesteps
        
        video_path = self.video_dir / f"step_{global_step:08d}.mp4"
        iio.imwrite(video_path, np.asarray(frames), fps=VIDEO_FPS, codec="libx264", quality=8)
        
        joints_path = self.joint_dir / f"step_{global_step:08d}.json"
        with joints_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "global_step": global_step,
                    "episode_return": total_reward,
                    "episode_length": len(frames),
                    "joints": joints_log,
                },
                f,
                indent=2,
            )
            
        # Log to TensorBoard
        self.logger.record("eval/episode_return", total_reward)
        self.logger.record("eval/episode_length", len(frames))
        # self.logger.record("eval/video_path", str(video_path))
        
        print(f"[video] step={global_step} -> {video_path}")

        # Plot losses
        self.plot_losses(global_step)

    def plot_losses(self, global_step: int):
        """Reads TensorBoard logs and plots loss curves."""
        log_dir = self.logger.get_dir()
        if not log_dir:
            return

        # Find event file
        event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        if not event_files:
            # If not in the logger dir, check parent dir (sometimes logger dir is subfolder)
            # But SB3 usually sets logger dir to specific algorithm folder
            return
        
        # Use the most recent one
        event_file = max(event_files, key=os.path.getmtime)
        
        try:
            # Load scalars
            ea = event_accumulator.EventAccumulator(
                event_file,
                size_guidance={event_accumulator.SCALARS: 0}, 
            )
            ea.Reload()
            
            tags = ea.Tags()["scalars"]
            # Filter for typical SAC losses and reward
            # train/actor_loss, train/critic_loss, train/ent_coef_loss, rollout/ep_rew_mean
            relevant_tags = [
                t for t in tags 
                if any(k in t for k in ["loss", "ep_rew_mean", "entropy"])
            ]
            
            if not relevant_tags:
                return

            # Plot
            n_plots = len(relevant_tags)
            cols = 2
            rows = (n_plots + 1) // cols
            if rows == 0: rows = 1
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
            if n_plots == 1:
                axes_list = [axes]
            else:
                axes_list = axes.flatten()
            
            for i, tag in enumerate(relevant_tags):
                data = ea.Scalars(tag)
                steps = [x.step for x in data]
                values = [x.value for x in data]
                
                ax = axes_list[i]
                ax.plot(steps, values)
                ax.set_title(tag)
                ax.set_xlabel("Step")
                ax.grid(True, alpha=0.3)
                
            # Hide unused subplots
            for j in range(i + 1, len(axes_list)):
                axes_list[j].axis('off')
                
            plt.tight_layout()
            plot_path = self.video_dir / f"loss_step_{global_step:08d}.png"
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"[plot] Loss graph saved to {plot_path}")
            
        except Exception as e:
            print(f"Warning: Failed to plot losses: {e}")


def main() -> None:
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    run_dir = Path(args.output_dir).expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Env
    # SAC typically uses 1 env, but we can use DummyVecEnv
    env = DummyVecEnv([make_env(args.seed, 0)])
    # Use VecNormalize for observation normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Callback
    video_callback = VideoAndJointLogCallback(
        check_freq=args.video_interval,
        log_dir=run_dir,
        seed=args.seed
    )
    
    # Initialize SAC
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=str(run_dir),
    )
    
    print(f"Training SAC on {ENV_ID} for {args.total_timesteps} steps...")
    model.learn(total_timesteps=args.total_timesteps, callback=video_callback, progress_bar=True)
    
    if args.save_policy:
        policy_path = run_dir / "policy"
        model.save(policy_path)
        print(f"Policy saved to {policy_path}")
        
        # Also save the VecNormalize statistics
        stats_path = run_dir / "vec_normalize.pkl"
        env.save(str(stats_path))
        print(f"VecNormalize stats saved to {stats_path}")

if __name__ == "__main__":
    main()
