#!/usr/bin/env python3
"""
Curriculum Learning PPO Agent for RedCubePick.
Automatically advances through stages: Reach -> Grasp -> Lift -> Place.
Uses Transformer Architecture for temporal memory.
"""

import argparse
import os
import random
import time
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import rl_env  # Registers env

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STAGE_REACH = 0
STAGE_GRASP = 1
STAGE_LIFT = 2
STAGE_PLACE = 3

STAGE_NAMES = {
    STAGE_REACH: "REACH",
    STAGE_GRASP: "GRASP",
    STAGE_LIFT: "LIFT",
    STAGE_PLACE: "PLACE"
}

# -----------------------------------------------------------------------------
# Curriculum Environment Wrapper
# -----------------------------------------------------------------------------
class CurriculumWrapper(gym.Wrapper):
    """Modifies reward based on current curriculum stage."""
    def __init__(self, env, stage=STAGE_REACH):
        super().__init__(env)
        self.stage = stage
        
    def set_stage(self, stage):
        self.stage = stage
        
    def step(self, action):
        obs, total_reward, term, trunc, info = self.env.step(action)
        
        # Extract raw components (env must return them in info)
        r_reach = info.get("reward_reach", 0.0)
        r_grasp = info.get("reward_grasp", 0.0)
        r_lift = info.get("reward_lift", 0.0)
        r_place = info.get("reward_place", 0.0)
        r_ctrl = info.get("reward_ctrl", 0.0)
        r_orient = info.get("reward_orientation", 0.0)
        
        # Recalculate Reward based on Stage
        reward = 0.0
        
        if self.stage == STAGE_REACH:
            # Focus purely on getting EE to cube
            reward = r_reach + r_orient + r_ctrl
            
        elif self.stage == STAGE_GRASP:
            # Focus on closing gripper near cube
            reward = r_reach + r_grasp * 2.0 + r_orient + r_ctrl
            
        elif self.stage == STAGE_LIFT:
            # Focus on lifting z > 0.05
            reward = r_reach + r_grasp + r_lift * 2.0 + r_orient + r_ctrl
            
        elif self.stage == STAGE_PLACE:
            # Full task
            reward = total_reward # Env default is good for full task
            
        return obs, reward, term, trunc, info

# -----------------------------------------------------------------------------
# Hyperparameters & Args
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Curriculum PPO Agent")
    parser.add_argument("--exp-name", type=str, default="ppo_curriculum", help="experiment name")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--total-timesteps", type=int, default=5000000, help="total timesteps")
    parser.add_argument("--num-envs", type=int, default=16, help="parallel envs")
    parser.add_argument("--video-interval", type=int, default=200000, help="video save interval")
    parser.add_argument("--start-stage", type=int, default=0, help="starting stage (0-3)")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max grad norm")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--update-epochs", type=int, default=4, help="update epochs")
    parser.add_argument("--cuda", type=lambda x: (str(x).lower() == 'true'), default=True, help="enable cuda")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * 128)
    args.minibatch_size = int(args.batch_size // 4)
    return args

ARGS = parse_args()

# -----------------------------------------------------------------------------
# Model (Transformer Architecture)
# -----------------------------------------------------------------------------
class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=np.inf, high=np.inf, shape=(k, shp[0]), dtype=env.observation_space.dtype)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k): self.frames.append(obs)
        return np.array(self.frames), info
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, term, trunc, info

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class TransformerAgent(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.obs_dim = obs_shape[1] # (K, Obs)
        self.act_dim = np.prod(action_shape)
        
        # Embedder
        self.embedding = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 256)), 
            nn.ReLU()
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=4, 
            dim_feedforward=1024, 
            dropout=0.0, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Heads
        self.actor_mean = layer_init(nn.Linear(256, self.act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim))
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_features(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # Pool: take last token
        return x[:, -1, :]

    def get_value(self, x):
        return self.critic(self.get_features(x))

    def get_action_and_value(self, x, action=None):
        feat = self.get_features(x)
        
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None: action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(feat)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def make_env(idx, run_name, stage):
    def thunk():
        env = gym.make("RedCubePick-v0", render_mode="rgb_array" if idx == 0 else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0:
            video_path = f"runs/{run_name}/videos"
            env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda x: x % 50 == 0)
        env = gym.wrappers.ClipAction(env)
        env = CurriculumWrapper(env, stage=stage)
        env = FrameStackWrapper(env, k=8)
        return env
    return thunk

def train():
    run_name = f"{ARGS.exp_name}__{ARGS.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() and ARGS.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Envs
    envs = gym.vector.SyncVectorEnv([make_env(i, run_name, ARGS.start_stage) for i in range(ARGS.num_envs)])
    
    # Agent
    agent = TransformerAgent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
    current_stage = ARGS.start_stage
    stage_success_history = deque(maxlen=100)
    
    # Storage
    obs = torch.zeros((128, ARGS.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((128, ARGS.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((128, ARGS.num_envs)).to(device)
    rewards = torch.zeros((128, ARGS.num_envs)).to(device)
    dones = torch.zeros((128, ARGS.num_envs)).to(device)
    values = torch.zeros((128, ARGS.num_envs)).to(device)
    
    next_obs, _ = envs.reset(seed=ARGS.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(ARGS.num_envs).to(device)
    
    print(f"Starting Curriculum Training at Stage {STAGE_NAMES[current_stage]}")
    
    global_step = 0
    num_updates = ARGS.total_timesteps // (128 * ARGS.num_envs)
    pbar = tqdm(range(1, num_updates + 1))
    
    dist_history = deque(maxlen=100)

    for update in pbar:
        # Anneal LR
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * 3e-4
        optimizer.param_groups[0]["lr"] = lrnow
        
        # Rollout
        for step in range(128):
            global_step += ARGS.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)
            
            # Track Success for Curriculum
            if "final_info" in info:
                for item in info["final_info"]:
                    if item:
                        if "dist_ee_cube" in item:
                            dist_history.append(item["dist_ee_cube"])

                        success = False
                        if current_stage == STAGE_REACH:
                            success = item["dist_ee_cube"] < 0.05 # Relaxed from 0.03
                        elif current_stage == STAGE_GRASP:
                            success = item["dist_ee_cube"] < 0.05 and item["gripper_cmd"] == 1
                        elif current_stage == STAGE_LIFT:
                            success = item["reward_lift"] > 0.5 # Using reward as proxy
                        elif current_stage == STAGE_PLACE:
                            success = item["is_success"]
                            
                        stage_success_history.append(1.0 if success else 0.0)
                        writer.add_scalar(f"curriculum/stage_{current_stage}_success", float(success), global_step)

        # Bootstrap
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(128)):
                if t == 128 - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + ARGS.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + ARGS.gamma * ARGS.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize
        b_inds = np.arange(ARGS.batch_size)
        for epoch in range(ARGS.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, ARGS.batch_size, ARGS.minibatch_size):
                end = start + ARGS.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - ARGS.clip_coef, 1 + ARGS.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -ARGS.clip_coef, ARGS.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = pg_loss - ARGS.ent_coef * entropy.mean() + v_loss * ARGS.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ARGS.max_grad_norm)
                optimizer.step()

        # Curriculum Logic
        success_rate = np.mean(stage_success_history) if len(stage_success_history) > 0 else 0.0
        avg_dist = np.mean(dist_history) if len(dist_history) > 0 else 0.0
        
        pbar.set_postfix({
            "stg": STAGE_NAMES[current_stage], 
            "succ": f"{success_rate:.2f}", 
            "rew": f"{rewards.mean().item():.2f}",
            "dist": f"{avg_dist:.3f}"
        })
        
        if len(stage_success_history) >= 50 and success_rate > 0.8 and current_stage < STAGE_PLACE:
            current_stage += 1
            print(f"\n🚀 ADVANCING TO STAGE {STAGE_NAMES[current_stage]} (Success Rate: {success_rate:.2f})")
            stage_success_history.clear()
            
            # Propagate stage to envs
            for e in envs.envs:
                curr = e
                while hasattr(curr, 'env'):
                    if isinstance(curr, CurriculumWrapper):
                        curr.set_stage(current_stage)
                        break
                    curr = curr.env
            
            writer.add_scalar("curriculum/stage", current_stage, global_step)
            
            # Save checkpoint at stage transition
            torch.save(agent.state_dict(), f"runs/{run_name}/policy_stage_{current_stage}.pt")

        # Save periodic checkpoint
        if update % (ARGS.video_interval // (128 * ARGS.num_envs)) == 0:
             torch.save(agent.state_dict(), f"runs/{run_name}/policy_step_{global_step}.pt")

    envs.close()
    writer.close()

if __name__ == "__main__":
    train()
