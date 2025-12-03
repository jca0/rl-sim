#!/usr/bin/env python3
"""
Learning from Demonstrations (LfD) Policy Training

Strategy:
1. Load human demos (teleop_dense.json)
2. Train a "waypoint tracker" that learns to move toward demo states
3. Use RL to fine-tune for robustness

This is MORE sample-efficient than pure RL because it starts from expert behavior.
"""

import argparse
import json
import time
from pathlib import Path
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import rl_env

# -----------------------------------------------------------------------------
# Load Demonstrations
# -----------------------------------------------------------------------------
def load_demos(demo_path="mujoco_sim/teleop_dense.json"):
    with open(demo_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    for key in sorted(data.keys()):
        if key.startswith("ACTION_SEQUENCE"):
            seq = np.array(data[key], dtype=np.float32)
            sequences.append(seq)
    
    print(f"Loaded {len(sequences)} demo sequences")
    return sequences

# -----------------------------------------------------------------------------
# Demo-Guided Reward Wrapper
# -----------------------------------------------------------------------------
class DemoGuidedWrapper(gym.Wrapper):
    """Adds a reward bonus for being close to demo trajectories."""
    def __init__(self, env, demos, bonus_weight=1.0):
        super().__init__(env)
        self.demos = demos
        self.bonus_weight = bonus_weight
        self.current_step = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        # Pick a random demo to follow
        self.current_demo = self.demos[np.random.randint(len(self.demos))]
        return obs, info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Get current joint positions
        current_joints = self.env.unwrapped.data.qpos[:6].copy()
        
        # Find closest waypoint in demo
        demo_len = len(self.current_demo)
        progress = min(self.current_step / 100.0, 1.0)  # Assume ~100 steps per demo
        demo_idx = int(progress * (demo_len - 1))
        target_joints = self.current_demo[demo_idx]
        
        # Compute distance to demo waypoint
        joint_dist = np.linalg.norm(current_joints - target_joints)
        
        # Bonus for being close to demo
        demo_bonus = np.exp(-5.0 * joint_dist) * self.bonus_weight
        
        reward += demo_bonus
        info["demo_bonus"] = demo_bonus
        info["joint_dist_to_demo"] = joint_dist
        
        self.current_step += 1
        return obs, reward, term, trunc, info

# -----------------------------------------------------------------------------
# Agent (Transformer)
# -----------------------------------------------------------------------------
class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(k, shp[0]), dtype=env.observation_space.dtype
        )
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
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
        self.obs_dim = obs_shape[1]
        self.act_dim = np.prod(action_shape)
        
        self.embedding = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 256)),
            nn.ReLU()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.actor_mean = layer_init(nn.Linear(256, self.act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim))
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_features(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x[:, -1, :]

    def get_value(self, x):
        return self.critic(self.get_features(x))

    def get_action_and_value(self, x, action=None):
        feat = self.get_features(x)
        action_mean = self.actor_mean(feat)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(feat)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="lfd_policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--demo-bonus", type=float, default=2.0, help="Weight for demo guidance")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == 'true', default=True)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * 128)
    args.minibatch_size = int(args.batch_size // 4)
    return args

ARGS = parse_args()

def make_env(idx, run_name, demos):
    def thunk():
        env = gym.make("RedCubePick-v0", render_mode="rgb_array" if idx == 0 else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"runs/{run_name}/videos", episode_trigger=lambda x: x % 50 == 0
            )
        env = gym.wrappers.ClipAction(env)
        env = DemoGuidedWrapper(env, demos, bonus_weight=ARGS.demo_bonus)
        env = FrameStackWrapper(env, k=8)
        return env
    return thunk

def train():
    run_name = f"{ARGS.exp_name}__{ARGS.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() and ARGS.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load demos
    demos = load_demos()
    
    # Create envs
    envs = gym.vector.SyncVectorEnv([make_env(i, run_name, demos) for i in range(ARGS.num_envs)])
    agent = TransformerAgent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
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
    
    global_step = 0
    num_updates = ARGS.total_timesteps // (128 * ARGS.num_envs)
    pbar = tqdm(range(1, num_updates + 1))
    
    success_history = deque(maxlen=100)
    
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
            
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "is_success" in item:
                        success_history.append(1.0 if item["is_success"] else 0.0)

        # Bootstrap + GAE
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
                delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize (4 epochs)
        b_inds = np.arange(ARGS.batch_size)
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, ARGS.batch_size, ARGS.minibatch_size):
                end = start + ARGS.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss - ARGS.ent_coef * entropy.mean() + v_loss * 0.5

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        success_rate = np.mean(success_history) if len(success_history) > 0 else 0.0
        pbar.set_postfix({"success": f"{success_rate:.2f}", "reward": f"{rewards.mean().item():.2f}"})
        
        if update % 50 == 0:
            torch.save(agent.state_dict(), f"runs/{run_name}/policy_step_{global_step}.pt")

    envs.close()
    writer.close()
    print(f"\n✅ Training complete! Policy saved to runs/{run_name}/")

if __name__ == "__main__":
    train()

