"""
Test if baseline policy works in optimized sim.

This will show us if the baseline policy fails in the optimized sim,
which would explain why fine-tuning is learning weird behavior.
"""

import sys
import json
from pathlib import Path
import torch
import gymnasium as gym
import imageio.v3 as iio
import mujoco

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels to SO-ARM100
sys.path.insert(0, str(REPO_ROOT / "submodule" / "rl-sim"))
sys.path.insert(0, str(REPO_ROOT / "working"))

import rl_env
from rl_env.train_trajectory_tracker import TrajectoryPolicy

# Load optimized parameters
OPTIMIZED_PARAMS_PATH = REPO_ROOT / "working" / "data" / "optimized_params.json"
with open(OPTIMIZED_PARAMS_PATH, "r") as f:
    opt_data = json.load(f)
opt_params = opt_data["optimized_parameters"]

print("\n" + "="*70)
print("TESTING BASELINE POLICY IN OPTIMIZED SIM")
print("="*70)

print("\nOptimized parameters:")
for param, value in opt_params.items():
    print(f"  {param:20s}: {value:.3f}")

# Create optimized environment
def create_optimized_env(params, render_mode=None):
    env = gym.make("RedCubePick-v0", render_mode=render_mode, max_episode_steps=600)
    
    # Apply optimized parameters
    for i in range(env.unwrapped.model.nu):
        actuator_name = env.unwrapped.model.actuator(i).name
        if "Jaw" in actuator_name:
            env.unwrapped.model.actuator_gainprm[i, 0] = params["gripper_kp"]
            env.unwrapped.model.actuator_biasprm[i, 1] = -params["gripper_kp"]
            env.unwrapped.model.actuator_forcerange[i, 0] = -params["gripper_forcerange"]
            env.unwrapped.model.actuator_forcerange[i, 1] = params["gripper_forcerange"]
        else:
            env.unwrapped.model.actuator_gainprm[i, 0] = params["arm_kp"]
            env.unwrapped.model.actuator_biasprm[i, 1] = -params["arm_kp"]
    
    for i in range(env.unwrapped.model.nv):
        env.unwrapped.model.dof_damping[i] = params["joint_damping"]
    
    for i in range(env.unwrapped.model.ngeom):
        geom_name = env.unwrapped.model.geom(i).name
        if "jaw_pad" in geom_name.lower():
            env.unwrapped.model.geom_friction[i, 0] = params["gripper_friction"]
    
    cube_body_id = env.unwrapped.cube_body_id
    env.unwrapped.model.body_mass[cube_body_id] = params["cube_mass"]
    
    return env

# Load baseline policy
BASELINE_POLICY_PATH = REPO_ROOT / "submodule" / "rl-sim" / "runs" / "FINAL_FIXED__1764557628" / "policy_epoch2000.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = TrajectoryPolicy()
policy.load_state_dict(torch.load(BASELINE_POLICY_PATH, map_location=device, weights_only=False))
policy.eval()
policy = policy.to(device)

print("\n✓ Baseline policy loaded")

# Test in DEFAULT sim
print("\n" + "="*70)
print("TEST 1: BASELINE IN DEFAULT SIM")
print("="*70)

env_default = gym.make("RedCubePick-v0", render_mode="rgb_array", max_episode_steps=600)
obs, _ = env_default.reset()
frames_default = []
reward_default = 0

for step in range(200):
    t_norm = step / 199.0
    t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        action = policy(t_tensor).cpu().numpy()[0]
    
    obs, reward, terminated, truncated, info = env_default.step(action)
    reward_default += reward
    frame = env_default.render()
    frames_default.append(frame)
    
    if terminated or truncated:
        break

env_default.close()

video_path_default = "baseline_in_DEFAULT_sim.mp4"
iio.imwrite(video_path_default, frames_default, fps=30)
print(f"\n✓ Video saved: {video_path_default}")
print(f"✓ Total reward: {reward_default:.2f}")
print(f"✓ Success: {info.get('is_success', False)}")

# Test in OPTIMIZED sim
print("\n" + "="*70)
print("TEST 2: BASELINE IN OPTIMIZED SIM")
print("="*70)

env_optimized = create_optimized_env(opt_params, render_mode="rgb_array")
obs, _ = env_optimized.reset()
frames_optimized = []
reward_optimized = 0

for step in range(200):
    t_norm = step / 199.0
    t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        action = policy(t_tensor).cpu().numpy()[0]
    
    obs, reward, terminated, truncated, info = env_optimized.step(action)
    reward_optimized += reward
    frame = env_optimized.render()
    frames_optimized.append(frame)
    
    if terminated or truncated:
        break

env_optimized.close()

video_path_optimized = "baseline_in_OPTIMIZED_sim.mp4"
iio.imwrite(video_path_optimized, frames_optimized, fps=30)
print(f"\n✓ Video saved: {video_path_optimized}")
print(f"✓ Total reward: {reward_optimized:.2f}")
print(f"✓ Success: {info.get('is_success', False)}")

# Summary
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Default sim reward:   {reward_default:.2f}")
print(f"Optimized sim reward: {reward_optimized:.2f}")
print(f"Difference:           {reward_optimized - reward_default:.2f}")

if reward_optimized < reward_default - 5.0:
    print("\n❌ BASELINE FAILS IN OPTIMIZED SIM!")
    print("   This explains why fine-tuning is learning weird behavior.")
    print("   The optimized parameters are TOO DIFFERENT from default.")
else:
    print("\n✅ Baseline works similarly in both sims")
    print("   Fine-tuning issue is something else (learning rate, etc.)")

print("\n🎬 Check the videos:")
print(f"  1. {video_path_default}")
print(f"  2. {video_path_optimized}")

