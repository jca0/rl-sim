#!/usr/bin/env python3
"""Vanilla Behavior Cloning trainer for RedCubePick using demonstrations from teleop.json."""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import rl_env  # registers the env

ENV_ID = "RedCubePick-v0"
VIDEO_FPS = 30
EVAL_MAX_STEPS = 600


class BCDataset(Dataset):
    """Dataset for behavior cloning."""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, obs):
        return self.net(obs)


def load_demonstrations(teleop_path: Path) -> List[List[List[float]]]:
    """Load action sequences from teleop.json."""
    with open(teleop_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    for key in sorted(data.keys()):
        if key.startswith("ACTION_SEQUENCE"):
            sequences.append(data[key])
    
    return sequences


def joint_pos_to_normalized_action(
    joint_pos: np.ndarray,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray
) -> np.ndarray:
    """Convert raw joint positions to normalized actions [-1, 1]."""
    normalized = np.zeros_like(joint_pos)
    
    # Convert arm joints (first 5)
    n_arm = len(joint_pos) - 1
    mid = 0.5 * (ctrl_low[:n_arm] + ctrl_high[:n_arm])
    range_val = ctrl_high[:n_arm] - ctrl_low[:n_arm]
    # Avoid division by zero
    range_val = np.where(range_val < 1e-6, 1.0, range_val)
    normalized[:n_arm] = 2.0 * (joint_pos[:n_arm] - mid) / range_val
    
    # Convert gripper (last joint)
    # Gripper: > 0 means close, <= 0 means open
    # Check if closer to high (close) or low (open)
    gripper_pos = joint_pos[-1]
    gripper_mid = 0.5 * (ctrl_low[-1] + ctrl_high[-1])
    if gripper_pos > gripper_mid:
        normalized[-1] = 0.5  # Close
    else:
        normalized[-1] = -0.5  # Open
    
    return np.clip(normalized, -1.0, 1.0)


def collect_demonstrations(
    env: gym.Env,
    sequences: List[List[List[float]]],
    seed: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Run demonstrations in environment and collect (obs, action) pairs."""
    observations = []
    actions = []
    
    env_unwrapped = env.unwrapped
    ctrl_low = env_unwrapped.ctrl_low
    ctrl_high = env_unwrapped.ctrl_high
    
    for seq_idx, sequence in enumerate(sequences):
        obs, _ = env.reset(seed=seed + seq_idx)
        
        for action_idx, action_raw in enumerate(sequence):
            # Convert raw joint positions to normalized actions
            joint_pos = np.array(action_raw, dtype=np.float32)
            
            # Convert to normalized action
            normalized_action = joint_pos_to_normalized_action(
                joint_pos, ctrl_low, ctrl_high
            )
            
            # Store (obs, action) pair BEFORE stepping
            observations.append(obs.copy())
            actions.append(normalized_action.copy())
            
            # Step environment
            obs, _, terminated, truncated, _ = env.step(normalized_action)
            
            # Continue even if terminated/truncated to collect all demonstration data
            # But break if we've exhausted the sequence
            if action_idx == len(sequence) - 1:
                break
    
    return np.array(observations), np.array(actions)


def train_bc(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: str,
    lr: float = 1e-3
) -> None:
    """Train behavior cloning policy."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            optimizer.zero_grad()
            pred_actions = model(obs_batch)
            loss = criterion(pred_actions, action_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")


def evaluate_policy(
    model: nn.Module,
    env: gym.Env,
    seed: int,
    max_steps: int = EVAL_MAX_STEPS,
    device: str = "cpu",
    min_steps: int = 30
) -> Tuple[List[np.ndarray], float]:
    """Evaluate policy and return frames and total reward."""
    model.eval()
    obs, _ = env.reset(seed=seed)
    
    frames = []
    total_reward = 0.0
    
    # Render initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame.copy())
    
    terminated = False
    truncated = False
    
    with torch.no_grad():
        for step in range(max_steps):
            # Predict action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = model(obs_tensor).cpu().numpy()[0]
            
            # Step environment
            obs, reward, term, trun, info = env.step(action)
            total_reward += float(reward)
            
            # Update termination flags
            terminated = terminated or bool(term)
            truncated = truncated or bool(trun)
            
            # Render after step
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())
            
            # Continue for at least min_steps to ensure we have a proper video
            if (terminated or truncated) and step >= min_steps:
                break
    
    print(f"Captured {len(frames)} frames, episode length: {step + 1}, terminated: {terminated}, truncated: {truncated}")
    if info:
        print(f"Info: {info}")
    return frames, total_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Behavior Cloning policy")
    parser.add_argument("--teleop-path", type=str, 
                       default="mujoco_sim/teleop.json",
                       help="Path to teleop.json with demonstrations")
    parser.add_argument("--exp-name", type=str, default="red_cube_pick_bc")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", 
                       help="cpu, cuda, or auto")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden dimension for MLP")
    parser.add_argument("--output-dir", type=str, default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    run_dir = Path(args.output_dir).expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Load demonstrations
    teleop_path = Path(args.teleop_path).expanduser()
    print(f"Loading demonstrations from {teleop_path}")
    sequences = load_demonstrations(teleop_path)
    print(f"Loaded {len(sequences)} demonstration sequences")
    
    # Create environment
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Collect demonstration data
    print("Collecting demonstration data...")
    observations, actions = collect_demonstrations(env, sequences, seed=args.seed)
    print(f"Collected {len(observations)} (obs, action) pairs")
    if len(observations) < 25:
        print(f"WARNING: Very few demonstration pairs collected ({len(observations)})!")
        print("This may indicate an issue with data collection.")
    
    # Create dataset and dataloader
    dataset = BCDataset(observations, actions)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # Create model
    model = BCPolicy(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    print(f"Training for {args.epochs} epochs...")
    train_bc(model, train_loader, args.epochs, device, lr=args.lr)
    
    # Save model
    model_path = run_dir / "bc_policy.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate and save video
    print("Evaluating policy...")
    eval_env = gym.make(ENV_ID, render_mode="rgb_array")
    frames, total_reward = evaluate_policy(
        model, eval_env, seed=args.seed, device=device
    )
    eval_env.close()
    
    if frames:
        video_path = video_dir / "final_policy.mp4"
        iio.imwrite(
            video_path, 
            np.asarray(frames), 
            fps=VIDEO_FPS, 
            codec="libx264", 
            quality=8
        )
        print(f"Video saved to {video_path}")
        print(f"Total reward: {total_reward:.2f}")
    else:
        print("Warning: No frames captured!")
    
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

