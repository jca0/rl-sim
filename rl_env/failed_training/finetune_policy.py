"""
Fine-tune the trained trajectory policy with optimized physics parameters.

This script:
1. Loads the baseline trained policy
2. Loads optimized parameters from parameter learning
3. Fine-tunes the policy for a small number of epochs (~50-100)
4. Uses lower learning rate to adapt to new physics
5. Saves fine-tuned policy for deployment
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "submodule" / "rl-sim"))

from train_trajectory_tracker import TrajectoryPolicy

# ============================================================================
# CONFIGURATION
# ============================================================================

BASELINE_POLICY_PATH = "runs/FINAL_FIXED__1764557628/policy_epoch2000.pt"
OPTIMIZED_PARAMS_PATH = str(REPO_ROOT / "working" / "data" / "optimized_params.json")
DEMO_PATH = "mujoco_sim/teleop.json"

# Fine-tuning hyperparameters
FINETUNE_EPOCHS = 100
LEARNING_RATE = 1e-5  # Much lower than initial training (was 3e-4)
BATCH_SIZE = 32
N_AUGMENT = 5  # Data augmentation

# Output
RUN_NAME = f"FINETUNED__{int(time.time())}"
OUTPUT_DIR = f"runs/{RUN_NAME}"

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("FINE-TUNE POLICY WITH OPTIMIZED PARAMETERS")
print("="*70)

# Load optimized parameters
print("\nLoading optimized parameters...")
with open(OPTIMIZED_PARAMS_PATH, "r") as f:
    opt_data = json.load(f)

optimized_params = opt_data["optimized_parameters"]
print("✓ Optimized parameters loaded:")
for param, value in optimized_params.items():
    print(f"  {param:20s}: {value:8.3f}")

# Load baseline policy
print("\nLoading baseline policy...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = TrajectoryPolicy().to(device)  # No arguments
policy.load_state_dict(torch.load(BASELINE_POLICY_PATH, map_location=device, weights_only=False))
print(f"✓ Baseline policy loaded from {BASELINE_POLICY_PATH}")
print(f"✓ Device: {device}")

# Load demonstration data
print("\nLoading demonstrations...")
with open(DEMO_PATH, "r") as f:
    demo_data = json.load(f)

# Extract demos
demo_sequences = []
for key, waypoints in demo_data.items():
    if key.startswith("ACTION_SEQUENCE_"):
        demo_sequences.append(waypoints)

print(f"✓ Loaded {len(demo_sequences)} demonstrations")

# ============================================================================
# CREATE ENVIRONMENT WITH OPTIMIZED PARAMETERS
# ============================================================================

print("\nCreating environment with optimized parameters...")

# TODO: In a full implementation, you would modify the MuJoCo XML here
# to apply the optimized parameters (gripper_kp, gripper_forcerange, etc.)
# For now, we'll use the base environment

env = gym.make("RedCubePick-v0", render_mode="rgb_array", max_episode_steps=250)
print("✓ Environment created")
print("⚠ Note: Parameter application to MuJoCo XML not yet implemented")
print("  In full pipeline, this would modify:")
for param in optimized_params.keys():
    print(f"    - {param}")

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================

print("\n" + "="*70)
print("PREPARING TRAINING DATA")
print("="*70)

def interpolate_demo(waypoints):
    """Interpolate demo waypoints to create dense trajectory."""
    JOINT_SPEED = 1.5
    MIN_MOVE_TIME = 0.2
    GRIPPER_TIME = 0.5
    dt = 0.002
    
    interpolated = []
    timestamps = []
    current_time = 0.0
    
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end = np.array(waypoints[i + 1])
        
        # Compute duration
        arm_dist = np.linalg.norm(end[:5] - start[:5])
        move_time = max(arm_dist / JOINT_SPEED, MIN_MOVE_TIME)
        
        gripper_changed = abs(end[5] - start[5]) > 0.1
        if gripper_changed:
            move_time += GRIPPER_TIME
        
        n_steps = int(np.ceil(move_time / dt))
        
        # Interpolate
        for step in range(n_steps):
            alpha = step / max(n_steps - 1, 1)
            interp = start + alpha * (end - start)
            interpolated.append(interp)
            timestamps.append(current_time + step * dt)
        
        current_time += n_steps * dt
    
    # Add final waypoint
    interpolated.append(np.array(waypoints[-1]))
    timestamps.append(current_time)
    
    return np.array(interpolated), np.array(timestamps)

# Process all demos
dataset = []

print("\nInterpolating demonstrations...")
for demo_idx, waypoints in enumerate(tqdm(demo_sequences, desc="Processing demos")):
    interpolated, timestamps = interpolate_demo(waypoints)
    
    # Normalize time to [0, 1]
    t_normalized = timestamps / timestamps[-1]
    
    # Add to dataset
    for t, joints in zip(t_normalized, interpolated):
        dataset.append((t, joints))

print(f"✓ Dataset size: {len(dataset)} samples")

# Data augmentation
print(f"\nApplying data augmentation (x{N_AUGMENT})...")
augmented_dataset = []

for t, joints in dataset:
    for _ in range(N_AUGMENT):
        # Time scaling
        t_aug = t + np.random.uniform(-0.02, 0.02)
        t_aug = np.clip(t_aug, 0.0, 1.0)
        
        # Joint noise
        joints_aug = joints + np.random.normal(0, 0.01, size=joints.shape)
        
        augmented_dataset.append((t_aug, joints_aug))

print(f"✓ Augmented dataset size: {len(augmented_dataset)} samples")

# Convert to tensors
t_data = torch.tensor([t for t, _ in augmented_dataset], dtype=torch.float32).unsqueeze(1).to(device)
joints_data = torch.tensor([j for _, j in augmented_dataset], dtype=torch.float32).to(device)

# ============================================================================
# FINE-TUNE POLICY
# ============================================================================

print("\n" + "="*70)
print("FINE-TUNING POLICY")
print("="*70)

optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/videos", exist_ok=True)

print(f"\nEpochs: {FINETUNE_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE} (baseline was 3e-4)")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Output: {OUTPUT_DIR}")

best_loss = float('inf')

for epoch in range(FINETUNE_EPOCHS):
    policy.train()
    
    # Shuffle data
    indices = torch.randperm(len(t_data))
    t_shuffled = t_data[indices]
    joints_shuffled = joints_data[indices]
    
    # Mini-batch training
    epoch_losses = []
    
    for i in range(0, len(t_data), BATCH_SIZE):
        batch_t = t_shuffled[i:i+BATCH_SIZE]
        batch_joints = joints_shuffled[i:i+BATCH_SIZE]
        
        # Forward pass
        pred_joints = policy(batch_t)
        loss = criterion(pred_joints, batch_joints)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    avg_loss = np.mean(epoch_losses)
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(policy.state_dict(), f"{OUTPUT_DIR}/policy_best.pt")
    
    # Periodic evaluation
    if epoch % 20 == 0 or epoch == FINETUNE_EPOCHS - 1:
        policy.eval()
        
        # Evaluate in environment
        obs, _ = env.reset()
        frames = []
        
        for step in range(250):
            t_normalized = min(step / 200.0, 1.0)
            
            with torch.no_grad():
                t_tensor = torch.tensor([[t_normalized]], dtype=torch.float32).to(device)
                action = policy(t_tensor).cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)
            
            if terminated or truncated:
                break
        
        # Save video
        import imageio
        video_path = f"{OUTPUT_DIR}/videos/eval_epoch{epoch:04d}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        
        success = info.get("is_success", False)
        print(f"Epoch {epoch:3d}/{FINETUNE_EPOCHS} | Loss: {avg_loss:.6f} | Success: {success}")
    
    elif epoch % 5 == 0:
        print(f"Epoch {epoch:3d}/{FINETUNE_EPOCHS} | Loss: {avg_loss:.6f}")

# Save final model
torch.save(policy.state_dict(), f"{OUTPUT_DIR}/policy_final.pt")

print("\n" + "="*70)
print("✅ FINE-TUNING COMPLETE!")
print("="*70)

print(f"\n✓ Models saved to {OUTPUT_DIR}/")
print(f"  - policy_best.pt (lowest loss)")
print(f"  - policy_final.pt (final epoch)")

print(f"\n✓ Videos saved to {OUTPUT_DIR}/videos/")

print("\n🎯 Next Steps:")
print("1. Record optimized sim trajectory:")
print("   python working/scripts/real_sim/record_sim_trajectory.py")
print("   (Update script to use fine-tuned policy)")
print("\n2. Deploy fine-tuned policy on real robot:")
print("   python working/scripts/real_sim/deploy_trajectory_policy.py")
print("   (Update script to use fine-tuned policy)")
print("\n3. Compare baseline vs optimized metrics!")

env.close()
print("\n" + "="*70)

