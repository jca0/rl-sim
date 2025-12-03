#!/usr/bin/env python3
"""
Trajectory Tracker - Learn to follow joint trajectories directly.

Key insight: Instead of learning state→action, learn time→joints.
This avoids compounding errors from state distribution shift.
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
# Dataset - Just (time, target_joints)
# -----------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """Dataset of (time, target_joints) from demos with PROPER TIMING."""
    def __init__(self, demo_path="mujoco_sim/teleop.json", augment=True):
        import math
        import mujoco
        from pathlib import Path
        
        self.samples = []
        self.augment = augment
        
        # Load MuJoCo model to get timestep
        scene_path = Path("mujoco_sim/trs_so_arm100/scene.xml")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        dt = model.opt.timestep
        
        JOINT_SPEED = 1.5
        MIN_MOVE_TIME = 0.2
        GRIPPER_TIME = 0.5
        
        with open(demo_path, 'r') as f:
            data = json.load(f)
        
        for key in sorted(data.keys()):
            if not key.startswith("ACTION_SEQUENCE"):
                continue
            
            sequence = np.array(data[key], dtype=np.float64)
            
            # Compute ACTUAL timing for each segment (same as visualize_demo.py)
            cumulative_time = 0.0
            time_stamps = [0.0]  # Start at t=0
            
            for i in range(len(sequence) - 1):
                start_pose = sequence[i]
                end_pose = sequence[i+1]
                
                # Determine duration
                joint_dist = np.linalg.norm(start_pose[:5] - end_pose[:5])
                gripper_dist = abs(start_pose[-1] - end_pose[-1])
                
                if joint_dist < 0.01 and gripper_dist > 0.1:
                    duration = GRIPPER_TIME
                else:
                    duration = max(joint_dist / JOINT_SPEED, MIN_MOVE_TIME)
                
                cumulative_time += duration
                time_stamps.append(cumulative_time)
            
            total_duration = time_stamps[-1]
            
            # Now create samples with interpolation
            for i in range(len(sequence) - 1):
                start_pose = sequence[i]
                end_pose = sequence[i+1]
                segment_start_time = time_stamps[i]
                segment_end_time = time_stamps[i+1]
                segment_duration = segment_end_time - segment_start_time
                
                # Sample points along this segment
                steps = int(math.ceil(segment_duration / dt))
                
                for s in range(0, steps, 3):  # Every 3rd step to avoid too much data
                    t = (s + 1) / steps
                    smooth_t = t * t * (3 - 2 * t)  # Smooth interpolation
                    
                    # Interpolated joint positions
                    joints = start_pose + (end_pose - start_pose) * smooth_t
                    
                    # Normalized time [0, 1]
                    time_norm = (segment_start_time + segment_duration * smooth_t) / total_duration
                    
                    self.samples.append({
                        'time': float(time_norm),
                        'joints': joints.astype(np.float32)
                    })
            
            # Add final waypoint
            self.samples.append({
                'time': 1.0,
                'joints': sequence[-1].astype(np.float32)
            })
        
        print(f"Collected {len(self.samples)} trajectory points from demos with proper timing")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor([sample['time']]),
            torch.FloatTensor(sample['joints'])
        )

# -----------------------------------------------------------------------------
# Policy - Time → Joints
# -----------------------------------------------------------------------------
class TrajectoryPolicy(nn.Module):
    """MLP that takes time and outputs target joint positions."""
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(1, 512),  # Just time as input
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),  # 6 joint positions
        )
    
    def forward(self, time):
        return self.net(time)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_policy(policy, device, num_episodes=5):
    """Evaluate policy by following time-based trajectory."""
    env = gym.make("RedCubePick-v0", render_mode="rgb_array")
    
    successes = []
    max_heights = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        step = 0
        max_steps = 300  # Run longer to see full trajectory
        demo_duration = 200  # Reference duration (demos take ~200 steps)
        max_height = 0.0
        
        frames = []
        
        # Don't terminate early - let full trajectory play out
        while step < max_steps:
            # Get target joints from policy based on time
            # Clamp time to [0, 1] so after demo completes, we hold final position
            time_norm = torch.FloatTensor([[min(step / demo_duration, 1.0)]]).to(device)
            
            with torch.no_grad():
                target_joints = policy(time_norm).cpu().numpy()[0]
            
            # Convert to action
            # Get current joints
            current_joints = env.unwrapped.data.qpos[:6].copy()
            
            # Compute action as normalized difference
            ctrl_low = env.unwrapped.ctrl_low
            ctrl_high = env.unwrapped.ctrl_high
            
            # Convert target joints to normalized action
            action = np.zeros(6, dtype=np.float32)
            action[:5] = 2.0 * (target_joints[:5] - ctrl_low[:5]) / (ctrl_high[:5] - ctrl_low[:5]) - 1.0
            action[:5] = np.clip(action[:5], -1.0, 1.0)
            
            # Gripper CONTINUOUS (not binary!) to preserve nuance
            action[5] = 2.0 * (target_joints[5] - ctrl_low[5]) / (ctrl_high[5] - ctrl_low[5]) - 1.0
            action[5] = np.clip(action[5], -1.0, 1.0)
            
            obs, reward, term, trunc, info = env.step(action)
            step += 1
            
            # Track cube height
            cube_height = env.unwrapped.data.xpos[env.unwrapped.cube_body_id][2]
            max_height = max(max_height, cube_height)
            
            if ep == 0:
                frames.append(env.render())
        
        successes.append(1.0 if info.get("is_success", False) else 0.0)
        max_heights.append(max_height)
        
        if ep == 0 and frames:
            return frames, np.mean(successes), np.mean(max_heights)
    
    env.close()
    return None, np.mean(successes), np.mean(max_heights)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="trajectory_tracker")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading demos...")
    dataset = TrajectoryDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create policy
    policy = TrajectoryPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    video_dir = Path(f"runs/{run_name}/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in tqdm(range(1, args.epochs + 1)):
        policy.train()
        epoch_loss = 0.0
        
        for time_norm, target_joints in dataloader:
            time_norm = time_norm.to(device)
            target_joints = target_joints.to(device)
            
            # Predict joints
            pred_joints = policy(time_norm)
            
            # MSE loss
            loss = nn.functional.mse_loss(pred_joints, target_joints)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        # Evaluate
        if epoch % args.eval_interval == 0:
            policy.eval()
            frames, success_rate, avg_height = evaluate_policy(policy, device)
            
            print(f"\nEpoch {epoch}: Loss={avg_loss:.6f}, Success={success_rate:.2f}, MaxHeight={avg_height:.4f}m")
            
            if frames:
                video_path = video_dir / f"eval_epoch{epoch:04d}.mp4"
                iio.imwrite(str(video_path), frames, fps=30)
                print(f"  Video saved: {video_path}")
            
            # Save checkpoint
            torch.save(policy.state_dict(), f"runs/{run_name}/policy_epoch{epoch}.pt")
    
    print(f"\n✅ Training complete! Policy saved to runs/{run_name}/")

if __name__ == "__main__":
    train()

