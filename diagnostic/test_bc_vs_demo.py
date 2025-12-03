#!/usr/bin/env python3
"""Test if BC policy actually replicates the demonstrations."""

import json
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import imageio.v3 as iio

import rl_env

# Load the BC policy
policy_path = Path("runs/promise__seed1__1764494541/act_policy.pt")  # Adjust if needed
teleop_path = Path("mujoco_sim/teleop.json")

print("Loading BC policy...")
env = gym.make("RedCubePick-v0", render_mode="rgb_array")

# Load ACT model architecture
from rl_env.train_cube_robot_env_act import ACTPolicy
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

model = ACTPolicy(obs_dim, action_dim, chunk_size=10, hidden_dim=256)
model.load_state_dict(torch.load(policy_path, map_location='cpu'))
model.eval()

print("Loading demo...")
with open(teleop_path) as f:
    data = json.load(f)

# Pick first demo
demo_key = "ACTION_SEQUENCE_0"
demo_joints = np.array(data[demo_key])

print(f"Demo: {demo_key}, {len(demo_joints)} waypoints")
print(f"Demo joints:\n{demo_joints}")

# Reset env to match demo start
obs, _ = env.reset(seed=1)

# Collect BC actions
bc_actions = []
frames = []

print("\nRunning BC policy...")
for step in range(200):
    frame = env.render()
    if frame is not None:
        frames.append(frame.copy())
    
    # Get BC action
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action_chunk = model(obs_tensor).cpu().numpy()[0]
        action = action_chunk[0]  # Take first action from chunk
    
    bc_actions.append(action.copy())
    
    # Step
    obs, reward, term, trun, info = env.step(action)
    
    if term or trun:
        print(f"Episode ended at step {step}")
        break

bc_actions = np.array(bc_actions)

print(f"\nBC Actions shape: {bc_actions.shape}")
print(f"BC Actions mean: {bc_actions.mean(axis=0)}")
print(f"BC Actions std: {bc_actions.std(axis=0)}")
print(f"\nFirst 5 BC actions:\n{bc_actions[:5]}")

# Save video
video_path = Path("bc_replay.mp4")
iio.imwrite(video_path, np.array(frames), fps=30, codec="libx264", quality=8)
print(f"\nSaved video: {video_path}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("1. Watch bc_replay.mp4 - does it look like the demo?")
print("2. Check if actions are reasonable (not all zeros, not exploding)")
print("3. Compare with demo actions visually")

env.close()

