#!/usr/bin/env python3
"""Action Chunking Transformer (ACT) trainer for RedCubePick."""

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
from tqdm.auto import tqdm, trange

import rl_env

ENV_ID = "RedCubePick-v0"
VIDEO_FPS = 30
EVAL_MAX_STEPS = 400  # ~13 seconds @ 30fps - more time to complete task


class ACTDataset(Dataset):
    """Dataset for ACT training with action chunking."""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        
        # Create valid indices (where we have enough future actions)
        self.valid_indices = []
        for i in range(len(observations) - chunk_size + 1):
            self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        obs = self.observations[start_idx]
        # Get next chunk_size actions
        action_chunk = self.actions[start_idx:start_idx + self.chunk_size]
        
        # Pad if necessary (shouldn't happen with valid_indices, but just in case)
        if len(action_chunk) < self.chunk_size:
            padding = torch.zeros(self.chunk_size - len(action_chunk), action_chunk.shape[1])
            action_chunk = torch.cat([action_chunk, padding], dim=0)
        
        return obs, action_chunk


class ACTPolicy(nn.Module):
    """Action Chunking Transformer policy."""
    
    def __init__(self, obs_dim: int, action_dim: int, chunk_size: int = 10, 
                 hidden_dim: int = 256, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        # Encode observation
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transformer for action sequence generation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Decode to action chunks
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size),
            nn.Tanh()
        )
    
    def forward(self, obs):
        # Encode observation
        obs_emb = self.obs_encoder(obs)  # (batch, hidden_dim)
        
        # Repeat for each chunk position
        obs_seq = obs_emb.unsqueeze(1).repeat(1, self.chunk_size, 1)  # (batch, chunk_size, hidden_dim)
        
        # Transform
        transformed = self.transformer(obs_seq)  # (batch, chunk_size, hidden_dim)
        
        # Take the aggregated representation (mean over sequence)
        aggregated = transformed.mean(dim=1)  # (batch, hidden_dim)
        
        # Decode to action chunk
        action_chunk = self.action_decoder(aggregated)  # (batch, action_dim * chunk_size)
        action_chunk = action_chunk.reshape(-1, self.chunk_size, self.action_dim)  # (batch, chunk_size, action_dim)
        
        return action_chunk


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
    mid = 0.5 * (ctrl_low + ctrl_high)
    range_val = ctrl_high - ctrl_low
    range_val = np.where(range_val < 1e-6, 1.0, range_val)
    normalized = 2.0 * (joint_pos - mid) / range_val
    return np.clip(normalized, -1.0, 1.0)


def collect_demonstrations_with_augmentation(
    env: gym.Env,
    sequences: List[List[List[float]]],
    seed: int = 1,
    n_augment: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect demos with data augmentation (add noise)."""
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
        # Noise level increases with augmentation index
        noise_scale = 0.02 * aug_idx  # 0, 0.02, 0.04
        
        for seq_idx, sequence in enumerate(sequences):
            obs, _ = env.reset(seed=seed + seq_idx + aug_idx * 1000)
            
            for i in range(len(sequence) - 1):
                start_pose = np.array(sequence[i], dtype=np.float32)
                end_pose = np.array(sequence[i+1], dtype=np.float32)
                
                # Add noise to arm joints only (not gripper)
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
                    
                    normalized_action = joint_pos_to_normalized_action(
                        target_pose, ctrl_low, ctrl_high
                    )
                    
                    observations.append(obs.copy())
                    actions.append(normalized_action.copy())
                    
                    obs, _, terminated, truncated, _ = env.step(normalized_action)
                    
                    if terminated or truncated:
                        break
                
                if terminated or truncated:
                    break
    
    return np.array(observations), np.array(actions)


def train_act(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: str,
    lr: float = 3e-4,
    eval_env: gym.Env = None,
    video_dir: Path = None,
    video_interval: int = 50,
    eval_seed: int = 42
) -> None:
    """Train ACT policy with periodic video evaluation."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Warmup for first 5% of training, then cosine decay
    warmup_epochs = max(int(0.05 * epochs), 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()
    
    pbar = trange(epochs, desc="Training ACT")
    
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for obs_batch, action_chunk_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_chunk_batch = action_chunk_batch.to(device)
            
            optimizer.zero_grad()
            pred_action_chunks = model(obs_batch)
            loss = criterion(pred_action_chunks, action_chunk_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{avg_loss:.6f}", "lr": f"{current_lr:.6f}"})
        
        # Save video at intervals
        if eval_env is not None and video_dir is not None and (epoch + 1) % video_interval == 0:
            pbar.write(f"[Epoch {epoch+1}] Saving video...")
            frames, reward, success = evaluate_policy(
                model, eval_env, seed=eval_seed, device=device
            )
            
            if frames:
                video_path = video_dir / f"training_epoch{epoch+1:04d}_reward{reward:.1f}.mp4"
                iio.imwrite(video_path, np.asarray(frames), fps=VIDEO_FPS, codec="libx264", quality=8)
                status = "✓ SUCCESS" if success else "✗ fail"
                pbar.write(f"  → {video_path.name} [{status}]")
    
    pbar.close()


def evaluate_policy(
    model: nn.Module,
    env: gym.Env,
    seed: int,
    max_steps: int = EVAL_MAX_STEPS,
    device: str = "cpu"
) -> Tuple[List[np.ndarray], float, bool]:
    """Evaluate ACT policy (executes only first action of chunk)."""
    model.eval()
    obs, _ = env.reset(seed=seed)
    
    frames = []
    total_reward = 0.0
    success = False
    
    frame = env.render()
    if frame is not None:
        frames.append(frame.copy())
    
    with torch.no_grad():
        for step in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action_chunk = model(obs_tensor).cpu().numpy()[0]
            
            # Execute only the first action in the chunk
            action = action_chunk[0]
            
            obs, reward, term, trun, info = env.step(action)
            total_reward += float(reward)
            
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())
            
            if info.get("is_success", False):
                success = True
            
            if term or trun:
                break
    
    return frames, total_reward, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train ACT policy")
    parser.add_argument("--teleop-path", type=str, default="mujoco_sim/teleop.json")
    parser.add_argument("--exp-name", type=str, default="act_baseline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=1000)  # More epochs for simple fitting
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4, help="Slightly higher for faster convergence")
    parser.add_argument("--chunk-size", type=int, default=10, help="Action chunk size")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--video-interval", type=int, default=50, help="Save video every N epochs")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Final eval episodes")
    parser.add_argument("--output-dir", type=str, default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print(f"Using device: {device}")
    
    # Create output directory
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir = run_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    
    # Load demonstrations
    teleop_path = Path(args.teleop_path)
    print(f"Loading from {teleop_path}")
    sequences = load_demonstrations(teleop_path)
    print(f"Loaded {len(sequences)} sequences")
    
    # Create environment
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Collect demos WITHOUT augmentation - just learn the exact demos
    observations, actions = collect_demonstrations_with_augmentation(
        env, sequences, seed=args.seed, n_augment=1  # No noise, just raw demos
    )
    print(f"Collected {len(observations)} training pairs")
    
    # Create dataset
    dataset = ACTDataset(observations, actions, chunk_size=args.chunk_size)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    model = ACTPolicy(
        obs_dim, action_dim, 
        chunk_size=args.chunk_size, 
        hidden_dim=args.hidden_dim
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create eval environment for periodic videos
    eval_env_for_training = gym.make(ENV_ID, render_mode="rgb_array")
    
    # Train
    print(f"Training for {args.epochs} epochs (videos every {args.video_interval} epochs)...")
    train_act(
        model, train_loader, args.epochs, device, 
        lr=args.lr,
        eval_env=eval_env_for_training,
        video_dir=video_dir,
        video_interval=args.video_interval,
        eval_seed=args.seed + 9999
    )
    
    eval_env_for_training.close()
    
    # Save model
    model_path = run_dir / "act_policy.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved to {model_path}")
    
    # Evaluate
    print(f"\nEvaluating {args.n_eval_episodes} episodes...")
    eval_env = gym.make(ENV_ID, render_mode="rgb_array")
    
    successes = 0
    for ep in range(args.n_eval_episodes):
        frames, reward, success = evaluate_policy(model, eval_env, seed=args.seed + ep, device=device)
        
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

