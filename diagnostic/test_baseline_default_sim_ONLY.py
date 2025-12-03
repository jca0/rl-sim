"""
Test ONLY the baseline policy in default sim.
This MUST work - if it doesn't, something is broken.
"""

import sys
from pathlib import Path
import torch
import gymnasium as gym
import imageio.v3 as iio

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodule" / "rl-sim"))

import rl_env
from rl_env.train_trajectory_tracker import TrajectoryPolicy

print("\n" + "="*70)
print("TESTING BASELINE POLICY IN DEFAULT SIM")
print("="*70)

# Load the EXACT policy that deploy script uses
POLICY_PATH = REPO_ROOT / "submodule" / "rl-sim" / "runs" / "FINAL_FIXED__1764557628" / "policy_epoch2000.pt"

print(f"\nLoading policy from:")
print(f"  {POLICY_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = TrajectoryPolicy()
policy.load_state_dict(torch.load(POLICY_PATH, map_location=device, weights_only=False))
policy.eval()
policy = policy.to(device)

print("✓ Policy loaded")

# Test in DEFAULT sim (no modifications)
print("\nRunning policy in DEFAULT sim...")
env = gym.make("RedCubePick-v0", render_mode="rgb_array", max_episode_steps=600)

print(f"Environment created:")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")

obs, _ = env.reset()
frames = []
total_reward = 0
cube_lifted = False
max_cube_height = 0

for step in range(200):
    # Get action from policy (time → joints)
    t_norm = step / 199.0
    t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        action = policy(t_tensor).cpu().numpy()[0]
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Track cube height
    cube_pos = env.unwrapped.data.xpos[env.unwrapped.cube_body_id]
    cube_height = cube_pos[2]
    max_cube_height = max(max_cube_height, cube_height)
    if cube_height > 0.05:
        cube_lifted = True
    
    # Render
    frame = env.render()
    frames.append(frame)
    
    if step % 50 == 0:
        print(f"  Step {step:3d}: reward={reward:.3f}, cube_height={cube_height:.3f}m")
    
    if terminated or truncated:
        print(f"  Episode ended at step {step}")
        break

env.close()

# Save video
video_path = "TEST_baseline_default_sim.mp4"
iio.imwrite(video_path, frames, fps=30)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Total reward: {total_reward:.2f}")
print(f"Max cube height: {max_cube_height:.3f}m")
print(f"Cube lifted (>5cm): {cube_lifted}")
print(f"Success: {info.get('is_success', False)}")
print(f"\n✓ Video saved: {video_path}")

print("\n" + "="*70)
if total_reward > 10.0 and cube_lifted:
    print("✅ BASELINE WORKS IN DEFAULT SIM!")
    print("   Policy is correct, ready for fine-tuning.")
elif total_reward > 5.0:
    print("⚠️  BASELINE PARTIALLY WORKS")
    print("   Gets some reward but may not complete task.")
else:
    print("❌ BASELINE FAILS IN DEFAULT SIM!")
    print("   Something is VERY wrong - this should work!")
    print("   Check:")
    print("   1. Policy file is correct")
    print("   2. Environment is not modified")
    print("   3. Action/observation spaces match")

print("="*70)

