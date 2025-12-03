#!/usr/bin/env python3
"""
Transformer-based PPO Agent for RedCubePick.
Combines PPO's reward optimization with a Transformer's ability to handle history/temporal dependencies.
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
from tqdm.auto import tqdm  # Add tqdm

import rl_env  # Registers env

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PPO Agent")
    parser.add_argument("--exp-name", type=str, default="ppo_transformer_memory",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--video-interval", type=int, default=200000,
        help="Save video every N steps")
        
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0)."""
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

# -----------------------------------------------------------------------------
# Constants (from args)
# -----------------------------------------------------------------------------
ARGS = parse_args()
EXP_NAME = ARGS.exp_name
SEED = ARGS.seed
CUDA = ARGS.cuda
TOTAL_TIMESTEPS = ARGS.total_timesteps
LEARNING_RATE = ARGS.learning_rate
NUM_ENVS = ARGS.num_envs
NUM_STEPS = ARGS.num_steps
MINIBATCH_SIZE = ARGS.minibatch_size
UPDATE_EPOCHS = ARGS.update_epochs
VIDEO_INTERVAL = ARGS.video_interval
CONTEXT_LENGTH = 8
HIDDEN_DIM = 256
N_HEADS = 4
N_LAYERS = 2
ENV_ID = "RedCubePick-v0"

# -----------------------------------------------------------------------------
# Frame Stacking Wrapper
# -----------------------------------------------------------------------------
class FrameStackWrapper(gym.Wrapper):
    """Stacks the last k observations."""
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.inf, high=np.inf, 
            shape=(k, shp[0]), 
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(self.frames)

# -----------------------------------------------------------------------------
# Transformer PPO Agent
# -----------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class TransformerAgent(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.obs_dim = obs_shape[1]  # (History, Obs_Dim)
        self.act_dim = np.prod(action_shape)
        
        # Feature Embedder
        self.embedding = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, HIDDEN_DIM)),
            nn.ReLU()
        )
        
        # Transformer Encoder (Processes History)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=N_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        
        # Heads
        self.actor_mean = layer_init(nn.Linear(HIDDEN_DIM, self.act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim))
        self.critic = layer_init(nn.Linear(HIDDEN_DIM, 1), std=1)

    def get_features(self, x):
        # x shape: (Batch, History, Obs_Dim)
        batch_size = x.shape[0]
        
        # Embed
        x = self.embedding(x)  # (Batch, History, Hidden)
        
        # Add simplified positional encoding (just a learnable vector added)
        # For simplicity in RL, often skipped if history is short, but let's rely on Transformer to figure out order
        
        # Pass through Transformer
        x = self.transformer(x)  # (Batch, History, Hidden)
        
        # Flatten / Pool: We take the LAST state (most recent) as the summary
        # because it has attended to all previous states
        return x[:, -1, :]  # (Batch, Hidden)

    def get_value(self, x):
        feat = self.get_features(x)
        return self.critic(feat)

    def get_action_and_value(self, x, action=None):
        feat = self.get_features(x)
        
        # Actor
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(feat)

# -----------------------------------------------------------------------------
# BC Pretraining Helpers
# -----------------------------------------------------------------------------
def load_demonstrations(teleop_path: str) -> List[List[List[float]]]:
    with open(teleop_path, 'r') as f:
        data = json.load(f)
    sequences = []
    for key in sorted(data.keys()):
        if key.startswith("ACTION_SEQUENCE"):
            sequences.append(data[key])
    return sequences

def joint_pos_to_normalized_action(joint_pos: np.ndarray, ctrl_low: np.ndarray, ctrl_high: np.ndarray) -> np.ndarray:
    mid = 0.5 * (ctrl_low + ctrl_high)
    range_val = ctrl_high - ctrl_low
    range_val = np.where(range_val < 1e-6, 1.0, range_val)
    normalized = 2.0 * (joint_pos - mid) / range_val
    
    # Force gripper (last dim) to be binary -1 or 1 to match env behavior
    # Env logic: > 0 is close, <= 0 is open
    # We want to teach the network to output strong signals
    # Check if current gripper pos is closer to high (close) or low (open)
    gripper_pos = joint_pos[-1]
    gripper_mid = mid[-1]
    
    if gripper_pos > gripper_mid:
        normalized[-1] = 1.0  # Close
    else:
        normalized[-1] = -1.0 # Open
        
    return np.clip(normalized, -1.0, 1.0)

def collect_bc_data(env, sequences, device, n_augment=5):
    """Replays demos in the env to collect (obs_stack, action) pairs with noise."""
    print(f"Collecting BC data from demonstrations ({n_augment}x augmentation)...")
    obs_list = []
    act_list = []
    
    # We need unwrapped access to control limits
    base_env = env.unwrapped
    ctrl_low = base_env.ctrl_low
    ctrl_high = base_env.ctrl_high
    dt = base_env.dt
    
    JOINT_SPEED = 1.5
    GRIPPER_TIME = 0.5
    MIN_MOVE_TIME = 0.2
    
    rng = np.random.default_rng(SEED)
    
    for aug_idx in range(n_augment):
        # Progressive noise: 0 (exact), then increasing
        noise_scale = 0.01 * aug_idx 
        
        for seq_idx, sequence in enumerate(sequences):
            # Reset with frame stack
            obs, _ = env.reset(seed=SEED + seq_idx + aug_idx * 1000)
            
            for i in range(len(sequence) - 1):
                start_pose = np.array(sequence[i], dtype=np.float32)
                end_pose = np.array(sequence[i+1], dtype=np.float32)
                
                # Add noise to arm joints only (first 5), not gripper
                if noise_scale > 0:
                    start_pose[:5] += rng.normal(0, noise_scale, 5)
                    end_pose[:5] += rng.normal(0, noise_scale, 5)
                
                joint_dist = np.linalg.norm(start_pose[:5] - end_pose[:5])
                gripper_dist = abs(start_pose[-1] - end_pose[-1])
                
                if joint_dist < 0.01 and gripper_dist > 0.1:
                    duration = GRIPPER_TIME
                else:
                    duration = max(joint_dist / JOINT_SPEED, MIN_MOVE_TIME)
                
                steps = max(int(duration / dt), 1)
                
                for s in range(steps):
                    alpha = (s + 1) / steps
                    target_pose = start_pose + (end_pose - start_pose) * alpha
                    
                    # Calculate action
                    norm_action = joint_pos_to_normalized_action(target_pose, ctrl_low, ctrl_high)
                    
                    # Store
                    obs_list.append(obs.copy())  # This is the stack (K, Obs)
                    act_list.append(norm_action.copy())
                    
                    # Step
                    obs, _, term, trunc, _ = env.step(norm_action)
                    if term or trunc:
                        break
                if term or trunc:
                    break
                
    return np.array(obs_list), np.array(act_list)

def pretrain_bc(agent, optimizer, device, history_len):
    """Pretrains the agent using BC."""
    teleop_path = "mujoco_sim/teleop.json"
    if not os.path.exists(teleop_path):
        print(f"Warning: {teleop_path} not found, skipping BC pretraining.")
        return

    # Create a dummy env for collection
    temp_env = make_env(ENV_ID, 0, False, "temp", history_len)()
    
    sequences = load_demonstrations(teleop_path)
    print(f"Loaded {len(sequences)} sequences for BC.")
    
    # Collect with augmentation
    obs_data, act_data = collect_bc_data(temp_env, sequences, device, n_augment=5)
    temp_env.close()
    
    print(f"Collected {len(obs_data)} samples. Starting BC pretraining (50 epochs)...")
    
    dataset = TensorDataset(torch.FloatTensor(obs_data).to(device), torch.FloatTensor(act_data).to(device))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    loss_fn = nn.MSELoss()
    
    for epoch in range(50):
        total_loss = 0
        for b_obs, b_act in loader:
            optimizer.zero_grad()
            feat = agent.get_features(b_obs)
            action_mean = agent.actor_mean(feat)
            loss = loss_fn(action_mean, b_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"  BC Epoch {epoch+1}: Loss {total_loss / len(loader):.4f}")
    
    print("BC Pretraining Complete.")

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def make_env(env_id, idx, capture_video, run_name, history_len):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if idx == 0 and capture_video else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0 and capture_video:
            video_path = f"runs/{run_name}/videos"
            env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda x: x % 50 == 0)
        env = gym.wrappers.ClipAction(env)
        env = FrameStackWrapper(env, k=history_len)
        return env
    return thunk

def train():
    run_name = f"{EXP_NAME}__seed{SEED}__{int(time.time())}"
    
    # Setup Writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars().items()])),
    )

    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = ARGS.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    print(f"Using device: {device}")

    # Env Setup
    # Use fixed run_name to ensure consistency
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, i == 0, run_name, CONTEXT_LENGTH) for i in range(NUM_ENVS)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Agent
    agent = TransformerAgent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # BC Pretraining (Leverage Action Sequences)
    pretrain_bc(agent, optimizer, device, CONTEXT_LENGTH)
    
    # Save BC-pretrained model
    bc_path = f"runs/{run_name}/bc_pretrained.pt"
    torch.save(agent.state_dict(), bc_path)
    print(f"Saved BC-pretrained model to {bc_path}")
    
    # Video Eval of BC
    print("Evaluating BC-pretrained policy...")
    eval_env_bc = make_env(ENV_ID, 0, True, run_name, CONTEXT_LENGTH)()
    obs, _ = eval_env_bc.reset(seed=SEED)
    # Need to fill the frame stack
    obs = torch.Tensor(obs).unsqueeze(0).to(device) # (1, K, Obs)
    
    for step in range(200):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        obs, _, _, _, _ = eval_env_bc.step(action.cpu().numpy()[0])
        obs = torch.Tensor(obs).unsqueeze(0).to(device)
    eval_env_bc.close()
    print("BC Evaluation Video Saved.")

    # Storage
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    
    print(f"Starting training for {TOTAL_TIMESTEPS} steps...")

    num_updates = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)
    
    pbar = tqdm(range(1, num_updates + 1), desc="Training PPO", unit="update")
    
    for update in pbar:
        # Annealing
        if ARGS.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout
        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # Execute
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        # Check custom success metric if available
                        if "is_success" in info:
                             writer.add_scalar("charts/success_rate", float(info["is_success"]), global_step)

        # Bootstrap
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
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

        # Optimization
        b_inds = np.arange(ARGS.batch_size)
        clipfracs = []
        
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, ARGS.batch_size, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > ARGS.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if ARGS.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - ARGS.clip_coef, 1 + ARGS.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if ARGS.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -ARGS.clip_coef,
                        ARGS.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ARGS.ent_coef * entropy_loss + v_loss * ARGS.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ARGS.max_grad_norm)
                optimizer.step()

            if ARGS.target_kl is not None and approx_kl > ARGS.target_kl:
                break

        # Logging
        if update % 10 == 0:
            reward_mean = rewards.mean().item()
            pbar.set_postfix({"reward": f"{reward_mean:.2f}", "loss": f"{loss.item():.2f}"})
            # print(f"Step {global_step}: reward={rewards.mean().item():.2f}, loss={loss.item():.2f}")
            
        if update % (VIDEO_INTERVAL // (NUM_STEPS * NUM_ENVS)) == 0:
             # Save Model
             save_path = f"runs/{run_name}/policy_step_{global_step}.pt"
             torch.save(agent.state_dict(), save_path)
             print(f"Saved policy to {save_path}")

    envs.close()
    writer.close()

if __name__ == "__main__":
    train()

