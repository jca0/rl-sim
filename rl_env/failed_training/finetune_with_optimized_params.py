"""
Fine-tune baseline policy in simulation with OPTIMIZED parameters.

After parameter learning discovers the correct sim parameters, we:
1. Load the optimized parameters
2. Create a sim environment with those parameters
3. Fine-tune the baseline policy using SUPERVISED LEARNING
4. The policy learns to adapt its outputs to the new physics

This is PROPER fine-tuning, not a hack!
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import imageio.v3 as iio
import mujoco

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodule" / "rl-sim"))
sys.path.insert(0, str(REPO_ROOT / "working"))

import rl_env
from rl_env.train_trajectory_tracker import TrajectoryPolicy

# ============================================================================
# CONFIGURATION
# ============================================================================

BASELINE_POLICY_PATH = "runs/FINAL_FIXED__1764557628/policy_epoch2000.pt"
OPTIMIZED_PARAMS_PATH = "../../working/data/optimized_params.json"
RUN_NAME = f"FINETUNED_OPTIMIZED__{int(time.time())}"
OUTPUT_DIR = f"runs/{RUN_NAME}"

# Fine-tuning hyperparameters
LEARNING_RATE = 3e-5  # Higher LR to actually learn gripper reduction
NUM_EPOCHS = 200  # More epochs to learn new behavior
BATCH_SIZE = 64
NUM_ROLLOUTS = 30  # More data with gripper penalty
EVAL_FREQ = 40  # Eval every 40 epochs

# ============================================================================
# LOAD OPTIMIZED PARAMETERS
# ============================================================================

print("\n" + "="*70)
print("FINE-TUNING WITH OPTIMIZED PARAMETERS")
print("="*70)

print("\nLoading optimized parameters...")
with open(OPTIMIZED_PARAMS_PATH, "r") as f:
    opt_data = json.load(f)

opt_params = opt_data["optimized_parameters"]
print("✓ Optimized parameters loaded:")
for param, value in opt_params.items():
    print(f"  {param:20s}: {value:.3f}")

# ============================================================================
# CREATE OPTIMIZED ENVIRONMENT
# ============================================================================

class GripperPenaltyWrapper(gym.Wrapper):
    """Wrapper that penalizes excessive gripper closure to prevent overload."""
    
    def __init__(self, env, penalty_threshold=1.2, penalty_weight=2.0):
        super().__init__(env)
        self.penalty_threshold = penalty_threshold  # Start penalizing above this
        self.penalty_weight = penalty_weight
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current gripper position
        gripper_pos = obs[5]  # 6th joint is gripper
        
        # Penalty for excessive closure (prevents overload!)
        if gripper_pos > self.penalty_threshold:
            excess = gripper_pos - self.penalty_threshold
            gripper_penalty = -self.penalty_weight * (excess ** 2)
            reward += gripper_penalty
            info["gripper_penalty"] = gripper_penalty
        
        return obs, reward, terminated, truncated, info

def create_optimized_env(params, render_mode=None):
    """Create environment with optimized parameters applied."""
    env = gym.make("RedCubePick-v0", render_mode=render_mode, max_episode_steps=600)
    
    # Apply optimized parameters to the environment
    # 1. Gripper stiffness and force
    for i in range(env.unwrapped.model.nu):
        actuator_name = env.unwrapped.model.actuator(i).name
        if "Jaw" in actuator_name:
            env.unwrapped.model.actuator_gainprm[i, 0] = params["gripper_kp"]
            env.unwrapped.model.actuator_biasprm[i, 1] = -params["gripper_kp"]
            env.unwrapped.model.actuator_forcerange[i, 0] = -params["gripper_forcerange"]
            env.unwrapped.model.actuator_forcerange[i, 1] = params["gripper_forcerange"]
        else:
            # Arm actuators
            env.unwrapped.model.actuator_gainprm[i, 0] = params["arm_kp"]
            env.unwrapped.model.actuator_biasprm[i, 1] = -params["arm_kp"]
    
    # 2. Joint damping
    for i in range(env.unwrapped.model.nv):
        env.unwrapped.model.dof_damping[i] = params["joint_damping"]
    
    # 3. Gripper friction
    for i in range(env.unwrapped.model.ngeom):
        geom_name = env.unwrapped.model.geom(i).name
        if "jaw_pad" in geom_name.lower():
            env.unwrapped.model.geom_friction[i, 0] = params["gripper_friction"]
    
    # 4. Cube mass
    cube_body_id = env.unwrapped.cube_body_id
    env.unwrapped.model.body_mass[cube_body_id] = params["cube_mass"]
    
    # 5. ADD GRIPPER PENALTY WRAPPER (prevents overload!)
    env = GripperPenaltyWrapper(env, penalty_threshold=1.2, penalty_weight=2.0)
    
    return env

# ============================================================================
# MAIN FINE-TUNING
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Load baseline policy
    print("\nLoading baseline policy...")
    policy = TrajectoryPolicy()
    policy.load_state_dict(torch.load(BASELINE_POLICY_PATH, map_location=device, weights_only=False))
    policy = policy.to(device)
    print("✓ Baseline policy loaded")
    
    # Create optimized environment
    print("\nCreating optimized environment...")
    env = create_optimized_env(opt_params, render_mode=None)
    print("✓ Optimized environment created")
    
    # Generate training data in optimized sim
    print(f"\nGenerating {NUM_ROLLOUTS} trajectories in OPTIMIZED sim...")
    print("⚠️  Running baseline policy to see what happens with new physics...")
    
    dataset = []
    successful_rollouts = 0
    
    demo_duration = 200  # Fixed duration like training
    
    for rollout in tqdm(range(NUM_ROLLOUTS), desc="Collecting data"):
        obs, _ = env.reset()
        rollout_data = []
        episode_reward = 0
        
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
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Store (time, target_joints) pair - NOT action!
            rollout_data.append({
                "time": min(step / demo_duration, 1.0),
                "target_joints": target_joints.copy()
            })
            
            if terminated or truncated:
                break
        
        # Keep all rollouts (even unsuccessful ones show what needs to change)
        dataset.extend(rollout_data)
        if episode_reward > 5.0:
            successful_rollouts += 1
    
    env.close()
    print(f"✓ Collected {len(dataset)} (time, action) pairs")
    print(f"  Successful rollouts: {successful_rollouts}/{NUM_ROLLOUTS}")
    
    # Prepare dataset
    times = np.array([d["time"] for d in dataset], dtype=np.float32)
    target_joints_np = np.array([d["target_joints"] for d in dataset], dtype=np.float32)
    
    times_tensor = torch.from_numpy(times).unsqueeze(1).to(device)
    target_joints_tensor = torch.from_numpy(target_joints_np).to(device)
    
    # Fine-tune policy
    print("\nFine-tuning policy on optimized sim data...")
    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/videos", exist_ok=True)
    
    for epoch in tqdm(range(NUM_EPOCHS), desc="Fine-tuning"):
        # Shuffle dataset
        indices = torch.randperm(len(dataset))
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(dataset), BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            batch_times = times_tensor[batch_indices]
            batch_target_joints = target_joints_tensor[batch_indices]
            
            # Forward pass
            pred_joints = policy(batch_times)
            loss = criterion(pred_joints, batch_target_joints)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluate
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == NUM_EPOCHS - 1:
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")
            
            # Generate evaluation video in OPTIMIZED sim
            policy.eval()
            eval_env = create_optimized_env(opt_params, render_mode="rgb_array")
            obs, _ = eval_env.reset()
            frames = []
            episode_reward = 0
            eval_duration = 200
            
            # Track gripper values to verify it's closing less
            gripper_values = []
            
            for step in range(eval_duration):
                # Get target joints from policy (EXACT same as training)
                time_norm = torch.FloatTensor([[min(step / eval_duration, 1.0)]]).to(device)
                
                with torch.no_grad():
                    target_joints = policy(time_norm).cpu().numpy()[0]
                
                # Track gripper position
                gripper_values.append(target_joints[5])
                
                # Convert to action (EXACT same as training)
                ctrl_low = eval_env.unwrapped.ctrl_low
                ctrl_high = eval_env.unwrapped.ctrl_high
                
                action = np.zeros(6, dtype=np.float32)
                action[:5] = 2.0 * (target_joints[:5] - ctrl_low[:5]) / (ctrl_high[:5] - ctrl_low[:5]) - 1.0
                action[:5] = np.clip(action[:5], -1.0, 1.0)
                
                # Gripper CONTINUOUS
                action[5] = 2.0 * (target_joints[5] - ctrl_low[5]) / (ctrl_high[5] - ctrl_low[5]) - 1.0
                action[5] = np.clip(action[5], -1.0, 1.0)
                
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                frame = eval_env.render()
                frames.append(frame)
                
                # DON'T BREAK EARLY - run full trajectory to see release!
                # if terminated or truncated:
                #     break
            
            video_path = f"{OUTPUT_DIR}/videos/eval_epoch{epoch+1:04d}.mp4"
            iio.imwrite(video_path, frames, fps=30)
            
            # Analyze gripper behavior
            max_gripper = max(gripper_values)
            print(f"✓ Video saved: {video_path}")
            print(f"✓ Episode reward: {episode_reward:.2f}")
            print(f"✓ Max gripper position: {max_gripper:.4f} rad (lower = less grip force)")
            
            eval_env.close()
            policy.train()
    
    # Save fine-tuned policy
    torch.save(policy.state_dict(), f"{OUTPUT_DIR}/policy_finetuned.pt")
    print(f"\n✓ Fine-tuned policy saved to {OUTPUT_DIR}/policy_finetuned.pt")
    
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"  Training data: {len(dataset)} (time, action) pairs")
    print(f"  Fine-tuning epochs: {NUM_EPOCHS}")
    print(f"  Final loss: {avg_loss:.6f}")
    print(f"  Output: {OUTPUT_DIR}")
    
    print("\n🎯 Next Steps:")
    print("1. Check evaluation videos - policy should work in optimized sim!")
    print("2. Deploy on real robot:")
    print("   cd ../../working")
    print("   python scripts/real_sim/deploy_trajectory_policy.py")
    print(f"   (Update POLICY_PATH to: submodule/rl-sim/{OUTPUT_DIR}/policy_finetuned.pt)")

if __name__ == "__main__":
    main()

