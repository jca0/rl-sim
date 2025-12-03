"""
Test baseline using EXACT same method as training evaluation.
This MUST work since it's the same code that generated the working eval videos.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import gymnasium as gym
import imageio.v3 as iio

# Setup
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodule" / "rl-sim"))

import rl_env
from rl_env.train_trajectory_tracker import TrajectoryPolicy

print("\n" + "="*70)
print("TESTING BASELINE - EXACT TRAINING EVALUATION METHOD")
print("="*70)

# Load policy
POLICY_PATH = REPO_ROOT / "submodule" / "rl-sim" / "runs" / "FINAL_FIXED__1764557628" / "policy_epoch2000.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = TrajectoryPolicy().to(device)
policy.load_state_dict(torch.load(POLICY_PATH, map_location=device, weights_only=False))
policy.eval()

print(f"✓ Policy loaded from {POLICY_PATH.name}")

# Create environment
env = gym.make("RedCubePick-v0", render_mode="rgb_array", max_episode_steps=600)
print("✓ Environment created")

# Evaluate (EXACT same code as train_trajectory_tracker.py lines 163-210)
obs, _ = env.reset()
frames = []
total_reward = 0
demo_duration = 200  # FIXED duration

print("\nRunning policy...")

for step in range(demo_duration):
    # Get target joints from policy (EXACT same as training)
    time_norm = torch.FloatTensor([[min(step / demo_duration, 1.0)]]).to(device)
    
    with torch.no_grad():
        target_joints = policy(time_norm).cpu().numpy()[0]
    
    # Convert to action (EXACT same as training)
    ctrl_low = env.unwrapped.ctrl_low
    ctrl_high = env.unwrapped.ctrl_high
    
    action = np.zeros(6, dtype=np.float32)
    action[:5] = 2.0 * (target_joints[:5] - ctrl_low[:5]) / (ctrl_high[:5] - ctrl_low[:5]) - 1.0
    action[:5] = np.clip(action[:5], -1.0, 1.0)
    
    # Gripper CONTINUOUS (not binary!)
    action[5] = 2.0 * (target_joints[5] - ctrl_low[5]) / (ctrl_high[5] - ctrl_low[5]) - 1.0
    action[5] = np.clip(action[5], -1.0, 1.0)
    
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    
    # Render
    frame = env.render()
    frames.append(frame)
    
    if step % 50 == 0:
        cube_height = env.unwrapped.data.xpos[env.unwrapped.cube_body_id][2]
        print(f"  Step {step:3d}: reward={reward:.3f}, cube_z={cube_height:.3f}m")

env.close()

# Save video
video_path = "TEST_baseline_EXACT_training_method.mp4"
iio.imwrite(video_path, frames, fps=30)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Total reward: {total_reward:.2f}")
print(f"Success: {info.get('is_success', False)}")
print(f"✓ Video saved: {video_path}")

if total_reward > 10.0:
    print("\n✅ BASELINE WORKS!")
    print("   This is the correct method - use this for fine-tuning!")
else:
    print("\n❌ BASELINE STILL FAILS")
    print("   Something is fundamentally broken.")

print("="*70)

