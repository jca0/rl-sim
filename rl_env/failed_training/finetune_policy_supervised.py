"""
Supervised fine-tuning of baseline policy in optimized simulation.

Strategy:
1. Load baseline policy (trained in default sim)
2. Run it in OPTIMIZED sim to see what joint positions it produces
3. Record (time, actual_joints_in_optimized_sim) pairs
4. Fine-tune policy to output those joint positions
5. This adapts the policy to the new physics (weaker gripper)

Key: We're teaching the policy "when you're at time t in the optimized sim,
you should output these joint positions (not the ones from default sim)"
"""

import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import imageio.v3 as iio

# Import optimized environment
import rl_env
from rl_env.cube_robot_env_optimized import RedCubePickEnvOptimized
from rl_env.train_trajectory_tracker import TrajectoryPolicy

# ============================================================================
# CONFIGURATION
# ============================================================================

BASELINE_POLICY_PATH = "runs/FINAL_FIXED__1764557628/policy_epoch2000.pt"
RUN_NAME = f"FINETUNED_SUPERVISED__{int(time.time())}"
OUTPUT_DIR = f"runs/{RUN_NAME}"

# Fine-tuning hyperparameters
LEARNING_RATE = 3e-5  # Moderate LR
NUM_EPOCHS = 200  # More epochs
BATCH_SIZE = 64
NUM_ROLLOUTS = 20  # Generate 20 trajectories in optimized sim
EVAL_FREQ = 20  # Evaluate every 20 epochs

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("SUPERVISED FINE-TUNING IN OPTIMIZED SIMULATION")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Load baseline policy
    print("\nLoading baseline policy...")
    policy = TrajectoryPolicy()
    policy.load_state_dict(torch.load(BASELINE_POLICY_PATH, map_location=device, weights_only=False))
    policy = policy.to(device)
    print(f"✓ Baseline policy loaded from {BASELINE_POLICY_PATH}")
    
    # Create optimized environment
    print("\nCreating optimized environment...")
    env = gym.make("RedCubePick-Optimized-v0", render_mode=None, max_episode_steps=600)
    print("✓ Optimized environment created")
    
    # Generate trajectories in optimized sim
    print(f"\nGenerating {NUM_ROLLOUTS} trajectories in optimized sim...")
    print("⚠️  Running baseline policy in OPTIMIZED sim to see what happens...")
    dataset = []
    successful_rollouts = 0
    
    for rollout in tqdm(range(NUM_ROLLOUTS), desc="Collecting data"):
        obs, _ = env.reset()
        rollout_data = []
        episode_reward = 0
        
        for step in range(200):  # 200 steps per trajectory
            t_norm = step / 199.0
            t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                action = policy(t_tensor).cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Store (time, action) pair
            rollout_data.append({
                "time": t_norm,
                "action": action.copy()
            })
            
            if terminated or truncated:
                break
        
        # Only keep successful rollouts (or high reward ones)
        if episode_reward > 5.0 or info.get("is_success", False):
            dataset.extend(rollout_data)
            successful_rollouts += 1
    
    env.close()
    print(f"✓ Collected {len(dataset)} (time, action) pairs from {successful_rollouts}/{NUM_ROLLOUTS} successful rollouts")
    
    # Prepare dataset
    times = torch.tensor([d["time"] for d in dataset], dtype=torch.float32, device=device).unsqueeze(1)
    actions = torch.tensor([d["action"] for d in dataset], dtype=torch.float32, device=device)
    
    # Fine-tune policy
    print("\nFine-tuning policy on optimized sim trajectories...")
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
            batch_times = times[batch_indices]
            batch_actions = actions[batch_indices]
            
            # Forward pass
            pred_actions = policy(batch_times)
            loss = criterion(pred_actions, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluate every EVAL_FREQ epochs
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == NUM_EPOCHS - 1:
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")
            
            # Generate evaluation video
            policy.eval()
            eval_env = gym.make("RedCubePick-Optimized-v0", render_mode="rgb_array", max_episode_steps=600)
            obs, _ = eval_env.reset()
            frames = []
            
            for step in range(200):
                t_norm = step / 199.0
                t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    action = policy(t_tensor).cpu().numpy()[0]
                
                obs, reward, terminated, truncated, info = eval_env.step(action)
                frame = eval_env.render()
                frames.append(frame)
                
                if terminated or truncated:
                    break
            
            video_path = f"{OUTPUT_DIR}/videos/eval_epoch{epoch+1:04d}.mp4"
            iio.imwrite(video_path, frames, fps=30)
            print(f"✓ Video saved: {video_path}")
            
            eval_env.close()
            policy.train()
    
    # Save fine-tuned policy
    torch.save(policy.state_dict(), f"{OUTPUT_DIR}/policy_finetuned.pt")
    print(f"\n✓ Fine-tuned policy saved to {OUTPUT_DIR}/policy_finetuned.pt")
    
    print("\n" + "="*70)
    print("✅ SUPERVISED FINE-TUNING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"  Training data: {len(dataset)} (time, action) pairs")
    print(f"  Fine-tuning epochs: {NUM_EPOCHS}")
    print(f"  Final loss: {avg_loss:.6f}")
    print(f"  Output: {OUTPUT_DIR}")
    
    print("\n🎯 Next Steps:")
    print("1. Check evaluation videos:")
    print(f"   {OUTPUT_DIR}/videos/")
    print("\n2. Deploy on real robot:")
    print("   Update POLICY_PATH in deploy_trajectory_policy.py")
    print(f"   to: submodule/rl-sim/{OUTPUT_DIR}/policy_finetuned.pt")

if __name__ == "__main__":
    main()

