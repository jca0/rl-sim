#!/usr/bin/env python3
"""LSTM-based BC trainer - better for temporal sequences."""

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

import rl_env

ENV_ID = "RedCubePick-v0"
VIDEO_FPS = 30
EVAL_MAX_STEPS = 200  # ~6.6 seconds @ 30fps


class LSTMDataset(Dataset):
    """Dataset with observation history for LSTM."""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray, history_len: int = 5):
        self.history_len = history_len
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        
        # Create valid indices (where we have enough history)
        self.valid_indices = list(range(history_len - 1, len(observations)))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        curr_idx = self.valid_indices[idx]
        # Get history
        start_idx = curr_idx - self.history_len + 1
        obs_history = self.observations[start_idx:curr_idx + 1]
        action = self.actions[curr_idx]
        
        return obs_history, action


class LSTMPolicy(nn.Module):
    """LSTM policy for behavior cloning."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs_history):
        # obs_history: (batch, history_len, obs_dim)
        lstm_out, _ = self.lstm(obs_history)
        # Take the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        action = self.fc(last_output)
        return action
    
    def predict(self, obs_history, hidden_state=None):
        """For inference with stateful hidden state."""
        with torch.no_grad():
            lstm_out, new_hidden = self.lstm(obs_history, hidden_state)
            last_output = lstm_out[:, -1, :]
            action = self.fc(last_output)
        return action, new_hidden


def load_demonstrations(teleop_path: Path) -> List[List[List[float]]]:
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
    return np.clip(normalized, -1.0, 1.0)


def collect_demonstrations_with_augmentation(
    env: gym.Env,
    sequences: List[List[List[float]]],
    seed: int = 1,
    n_augment: int = 5  # More augmentation for limited data
) -> Tuple[np.ndarray, np.ndarray]:
    observations = []
    actions = []
    
    env_unwrapped = env.unwrapped
    ctrl_low = env_unwrapped.ctrl_low
    ctrl_high = env_unwrapped.ctrl_high
    dt = env_unwrapped.dt
    
    JOINT_SPEED = 1.5
    GRIPPER_TIME = 0.5
    MIN_MOVE_TIME = 0.2
    
    rng = np.random.default_rng(seed)
    
    print(f"Collecting data with {n_augment}x augmentation...")
    
    for aug_idx in range(n_augment):
        noise_scale = 0.015 * aug_idx  # Progressive noise
        
        for seq_idx, sequence in enumerate(sequences):
            obs, _ = env.reset(seed=seed + seq_idx + aug_idx * 1000)
            
            for i in range(len(sequence) - 1):
                start_pose = np.array(sequence[i], dtype=np.float32)
                end_pose = np.array(sequence[i+1], dtype=np.float32)
                
                if noise_scale > 0:
                    start_pose[:5] += rng.normal(0, noise_scale, 5)
                    end_pose[:5] += rng.normal(0, noise_scale, 5)
                
                joint_dist = np.linalg.norm(start_pose[:5] - end_pose[:5])
                gripper_dist = abs(start_pose[-1] - end_pose[-1])
                
                duration = GRIPPER_TIME if (joint_dist < 0.01 and gripper_dist > 0.1) else max(joint_dist / JOINT_SPEED, MIN_MOVE_TIME)
                steps = max(int(duration / dt), 1)
                
                for s in range(steps):
                    alpha = (s + 1) / steps
                    target_pose = start_pose + (end_pose - start_pose) * alpha
                    normalized_action = joint_pos_to_normalized_action(target_pose, ctrl_low, ctrl_high)
                    
                    observations.append(obs.copy())
                    actions.append(normalized_action.copy())
                    
                    obs, _, terminated, truncated, _ = env.step(normalized_action)
                    if terminated or truncated:
                        break
                
                if terminated or truncated:
                    break
    
    return np.array(observations), np.array(actions)


def train_lstm(model: nn.Module, train_loader: DataLoader, epochs: int, device: str, lr: float = 1e-3) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    history_len: int,
    max_steps: int = EVAL_MAX_STEPS,
    device: str = "cpu"
) -> Tuple[List[np.ndarray], float, bool]:
    model.eval()
    obs, _ = env.reset(seed=seed)
    
    frames = []
    total_reward = 0.0
    success = False
    
    # Initialize history with repeated first observation
    obs_history = [obs.copy() for _ in range(history_len)]
    
    frame = env.render()
    if frame is not None:
        frames.append(frame.copy())
    
    with torch.no_grad():
        for step in range(max_steps):
            # Prepare history tensor
            obs_tensor = torch.FloatTensor(np.array(obs_history)).unsqueeze(0).to(device)
            action = model(obs_tensor).cpu().numpy()[0]
            
            obs, reward, term, trun, info = env.step(action)
            total_reward += float(reward)
            
            # Update history
            obs_history.pop(0)
            obs_history.append(obs.copy())
            
            if info.get("is_success", False):
                success = True
            
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())
            
            if term or trun:
                break
    
    return frames, total_reward, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train LSTM policy")
    parser.add_argument("--teleop-path", type=str, default="mujoco_sim/teleop.json")
    parser.add_argument("--exp-name", type=str, default="lstm_baseline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print(f"Using device: {device}")
    
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir = run_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    
    teleop_path = Path(args.teleop_path)
    print(f"Loading from {teleop_path}")
    sequences = load_demonstrations(teleop_path)
    print(f"Loaded {len(sequences)} sequences")
    
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}, History len: {args.history_len}")
    
    observations, actions = collect_demonstrations_with_augmentation(env, sequences, seed=args.seed, n_augment=5)
    print(f"Collected {len(observations)} training pairs")
    
    dataset = LSTMDataset(observations, actions, history_len=args.history_len)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = LSTMPolicy(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print(f"Training for {args.epochs} epochs...")
    train_lstm(model, train_loader, args.epochs, device, lr=args.lr)
    
    model_path = run_dir / "lstm_policy.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved to {model_path}")
    
    print(f"\nEvaluating {args.n_eval_episodes} episodes...")
    eval_env = gym.make(ENV_ID, render_mode="rgb_array")
    
    successes = 0
    for ep in range(args.n_eval_episodes):
        frames, reward, success = evaluate_policy(
            model, eval_env, seed=args.seed + ep, history_len=args.history_len, device=device
        )
        
        if success:
            successes += 1
        
        if frames:
            video_path = video_dir / f"eval_ep{ep}_reward{reward:.1f}.mp4"
            iio.imwrite(video_path, np.asarray(frames), fps=VIDEO_FPS, codec="libx264", quality=8)
            print(f"  Ep {ep}: Reward={reward:.2f}, Success={success}, Video={video_path.name}")
    
    print(f"\nSuccess rate: {successes}/{args.n_eval_episodes} ({100*successes/args.n_eval_episodes:.0f}%)")
    
    eval_env.close()
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

