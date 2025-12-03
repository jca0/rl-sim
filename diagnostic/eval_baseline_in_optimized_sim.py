"""
Evaluate baseline policy in OPTIMIZED simulation.

Key Question: Does the baseline policy work in the optimized sim?
If YES: We can deploy it directly!
If NO: We need fine-tuning.
"""

import gymnasium as gym
import torch
import imageio.v3 as iio
from pathlib import Path

import rl_env
from rl_env.cube_robot_env_optimized import RedCubePickEnvOptimized
from rl_env.train_trajectory_tracker import TrajectoryPolicy

# Load baseline policy
BASELINE_POLICY_PATH = "runs/FINAL_FIXED__1764557628/policy_epoch2000.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "="*70)
print("TESTING BASELINE POLICY IN OPTIMIZED SIMULATION")
print("="*70)

print(f"\nLoading baseline policy from {BASELINE_POLICY_PATH}...")
policy = TrajectoryPolicy()
policy.load_state_dict(torch.load(BASELINE_POLICY_PATH, map_location=device, weights_only=False))
policy = policy.to(device)
policy.eval()
print("✓ Baseline policy loaded")

# Test in optimized sim
print("\nTesting in OPTIMIZED sim (weaker gripper: 21.3 N)...")
env = gym.make("RedCubePick-Optimized-v0", render_mode="rgb_array", max_episode_steps=600)

NUM_EPISODES = 5
successes = 0
rewards = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    frames = []
    episode_reward = 0
    
    print(f"\nEpisode {ep+1}/{NUM_EPISODES}:")
    
    for step in range(200):
        t_norm = step / 199.0
        t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            action = policy(t_tensor).cpu().numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if ep == 0:  # Record first episode
            frame = env.render()
            frames.append(frame)
        
        if terminated or truncated:
            break
    
    is_success = info.get("is_success", False)
    if is_success:
        successes += 1
    
    rewards.append(episode_reward)
    print(f"  Reward: {episode_reward:.2f}, Success: {is_success}")
    
    # Save video of first episode
    if ep == 0:
        video_path = "baseline_in_optimized_sim.mp4"
        iio.imwrite(video_path, frames, fps=30)
        print(f"\n✓ Video saved: {video_path}")

env.close()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Success Rate: {successes}/{NUM_EPISODES} = {successes/NUM_EPISODES*100:.0f}%")
print(f"Average Reward: {sum(rewards)/len(rewards):.2f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if successes >= 3:
    print("✅ BASELINE WORKS IN OPTIMIZED SIM!")
    print("\n🎯 Next Step: Deploy baseline directly on real robot!")
    print("   The optimized sim parameters should prevent overload!")
else:
    print("❌ BASELINE DOESN'T WORK IN OPTIMIZED SIM")
    print("\n🔧 Next Step: Fine-tune policy in optimized sim")
    print("   Need to adapt to weaker gripper force")

print("\n📹 Check the video: baseline_in_optimized_sim.mp4")

