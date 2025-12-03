#!/usr/bin/env python3
"""
Direct Imitation Learning - Pure Supervised Learning from Demos

NO RL. Just train a policy to predict: given current state + time, what action should I take?

This should work because:
1. We have 50 successful demos
2. We know the exact state-action pairs
3. We just need to memorize and generalize slightly
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import imageio.v3 as iio

import rl_env

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class DemoDataset(Dataset):
    """Dataset of (state, time, action) tuples from demos."""
    def __init__(self, demo_path="mujoco_sim/teleop.json"):
        self.samples = []
        
        # Load demos
        with open(demo_path, 'r') as f:
            data = json.load(f)
        
        # Create environment to get states
        env = gym.make("RedCubePick-v0")
        
        for key in sorted(data.keys()):
            if not key.startswith("ACTION_SEQUENCE"):
                continue
            
            sequence = np.array(data[key], dtype=np.float32)
            
            # Reset env
            obs, _ = env.reset()
            
            # Play through sequence and collect (state, time, action) pairs
            for t, waypoint in enumerate(sequence):
                # Current state
                state = obs.copy()
                
                # Time progress (normalized)
                time_progress = t / len(sequence)
                
                # Target action (convert joint pos to normalized action)
                action = self._joint_to_action(waypoint, env.unwrapped)
                
                self.samples.append({
                    'state': state,
                    'time': time_progress,
                    'action': action
                })
                
                # Step env to get next state
                obs, _, _, _, _ = env.step(action)
        
        env.close()
        print(f"Collected {len(self.samples)} state-action pairs from demos")
    
    def _joint_to_action(self, joint_pos, env):
        """Convert joint positions to normalized actions."""
        ctrl_low = env.ctrl_low
        ctrl_high = env.ctrl_high
        
        action = np.zeros(6, dtype=np.float32)
        
        # Arm (continuous)
        action[:5] = 2.0 * (joint_pos[:5] - ctrl_low[:5]) / (ctrl_high[:5] - ctrl_low[:5]) - 1.0
        action[:5] = np.clip(action[:5], -1.0, 1.0)
        
        # Gripper (binary)
        gripper_mid = (ctrl_low[5] + ctrl_high[5]) / 2.0
        action[5] = 1.0 if joint_pos[5] > gripper_mid else -1.0
        
        return action
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['state']),
            torch.FloatTensor([sample['time']]),
            torch.FloatTensor(sample['action'])
        )

# -----------------------------------------------------------------------------
# Policy Network
# -----------------------------------------------------------------------------
class ImitationPolicy(nn.Module):
    """Simple MLP that takes (state, time) and outputs action."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 512),  # +1 for time
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, state, time):
        x = torch.cat([state, time], dim=-1)
        return self.net(x)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="direct_imitation")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=50)
    return parser.parse_args()

def evaluate_policy(policy, device, num_episodes=5):
    """Evaluate policy in simulation."""
    env = gym.make("RedCubePick-v0", render_mode="rgb_array")
    
    successes = []
    max_heights = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        max_height = 0.0
        
        frames = []
        
        while not done and step < 600:
            # Get action from policy
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            time_progress = torch.FloatTensor([[step / 600.0]]).to(device)
            
            with torch.no_grad():
                action = policy(state, time_progress).cpu().numpy()[0]
            
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1
            
            # Track cube height
            cube_height = env.unwrapped.data.xpos[env.unwrapped.cube_body_id][2]
            max_height = max(max_height, cube_height)
            
            if ep == 0:  # Save video of first episode
                frames.append(env.render())
        
        successes.append(1.0 if info.get("is_success", False) else 0.0)
        max_heights.append(max_height)
        
        if ep == 0 and frames:
            return frames, np.mean(successes), np.mean(max_heights)
    
    env.close()
    return None, np.mean(successes), np.mean(max_heights)

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading demos...")
    dataset = DemoDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create policy
    state_dim = 21  # RedCubePick observation dim
    action_dim = 6
    policy = ImitationPolicy(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    video_dir = Path(f"runs/{run_name}/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in tqdm(range(1, args.epochs + 1)):
        policy.train()
        epoch_loss = 0.0
        
        for state, time_prog, action in dataloader:
            state = state.to(device)
            time_prog = time_prog.to(device)
            action = action.to(device)
            
            # Predict action
            pred_action = policy(state, time_prog)
            
            # MSE loss
            loss = nn.functional.mse_loss(pred_action, action)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        # Evaluate
        if epoch % args.eval_interval == 0:
            policy.eval()
            frames, success_rate, avg_height = evaluate_policy(policy, device)
            
            print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Success={success_rate:.2f}, MaxHeight={avg_height:.4f}m")
            
            if frames:
                video_path = video_dir / f"eval_epoch{epoch:04d}.mp4"
                iio.imwrite(str(video_path), frames, fps=30)
                print(f"  Video saved: {video_path}")
            
            # Save checkpoint
            torch.save(policy.state_dict(), f"runs/{run_name}/policy_epoch{epoch}.pt")
    
    print(f"\n✅ Training complete! Policy saved to runs/{run_name}/")

if __name__ == "__main__":
    train()

