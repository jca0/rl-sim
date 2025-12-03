#!/usr/bin/env python3
"""
Evaluate the final trained policy with extended time.
"""

import sys
import torch
import numpy as np
import gymnasium as gym
import imageio.v3 as iio
from pathlib import Path

import rl_env

# Import the policy class
from rl_env.train_trajectory_tracker import TrajectoryPolicy

def evaluate(policy_path, num_episodes=5, max_steps=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load policy
    policy = TrajectoryPolicy().to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    
    print(f"Loaded policy from {policy_path}")
    print(f"Evaluating for {num_episodes} episodes with max {max_steps} steps each...")
    
    env = gym.make("RedCubePick-v0", render_mode="rgb_array")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        step = 0
        
        frames = []
        
        # Don't stop early - run for full trajectory
        while step < max_steps:
            # Get target joints from policy based on time
            time_norm = torch.FloatTensor([[step / max_steps]]).to(device)
            
            with torch.no_grad():
                target_joints = policy(time_norm).cpu().numpy()[0]
            
            # Convert to action
            current_joints = env.unwrapped.data.qpos[:6].copy()
            ctrl_low = env.unwrapped.ctrl_low
            ctrl_high = env.unwrapped.ctrl_high
            
            action = np.zeros(6, dtype=np.float32)
            action[:5] = 2.0 * (target_joints[:5] - ctrl_low[:5]) / (ctrl_high[:5] - ctrl_low[:5]) - 1.0
            action[:5] = np.clip(action[:5], -1.0, 1.0)
            
            gripper_mid = (ctrl_low[5] + ctrl_high[5]) / 2.0
            action[5] = 1.0 if target_joints[5] > gripper_mid else -1.0
            
            obs, reward, term, trunc, info = env.step(action)
            step += 1
            
            frames.append(env.render())
        
        # Save video
        cube_height = env.unwrapped.data.xpos[env.unwrapped.cube_body_id][2]
        success = info.get("is_success", False)
        
        output_path = Path(f"runs/final_eval_ep{ep}_success{success}_height{cube_height:.3f}.mp4")
        iio.imwrite(str(output_path), frames, fps=30)
        
        print(f"Episode {ep}: Success={success}, Height={cube_height:.4f}m, Steps={step}, Video={output_path}")
    
    env.close()
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_final_policy.py <path_to_policy.pt>")
        print("Example: python eval_final_policy.py runs/FINAL_working_demos__1764555948/policy_epoch2000.pt")
        sys.exit(1)
    
    policy_path = sys.argv[1]
    evaluate(policy_path)

