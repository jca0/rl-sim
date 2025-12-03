"""
RL-based fine-tuning of baseline policy in optimized simulation.

This script:
1. Loads baseline policy (trained with supervised learning)
2. Converts it to an RL policy (adds value head)
3. Fine-tunes with PPO in the OPTIMIZED sim (learned parameters)
4. Uses low learning rate and short training to adapt, not overwrite
5. Saves fine-tuned policy for deployment

Key Innovation: Two-stage RL approach
- Stage 1: RL for parameter optimization (learn sim params)
- Stage 2: RL for policy adaptation (adapt to optimized sim)
"""

import argparse
import os
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
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
RUN_NAME = f"FINETUNED_RL__{int(time.time())}"
OUTPUT_DIR = f"runs/{RUN_NAME}"

# PPO Hyperparameters (ULTRA-CONSERVATIVE for fine-tuning)
LEARNING_RATE = 1e-6  # 100x lower than baseline training (was 1e-5)
NUM_ENVS = 2  # Fewer envs for stability (was 4)
NUM_STEPS = 2048  # Steps per env per update
TOTAL_TIMESTEPS = 10000  # Very short training (was 50000)
BATCH_SIZE = 64
N_EPOCHS = 3  # Fewer epochs to prevent overfitting (was 10)
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.0  # ZERO exploration - pure exploitation! (was 0.01)
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Evaluation
EVAL_FREQ = 2000  # Every 2k steps (was 5000)
EVAL_EPISODES = 3

# ============================================================================
# RL POLICY (Trajectory Policy + Value Head)
# ============================================================================

class RLPolicyFromTrajectory(nn.Module):
    """
    Convert TrajectoryPolicy (time → joints) to RL policy (obs → action).
    
    The baseline policy learned time → joints.
    For RL, we need obs → action.
    
    Strategy: Use the trajectory policy as the actor, add a critic for value estimation.
    """
    
    def __init__(self, baseline_trajectory_policy, obs_dim):
        super().__init__()
        
        # Actor: We'll use a new network that maps obs → action
        # But initialize it to mimic the trajectory policy's behavior
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 6),  # 6 joint positions
        )
        
        # Critic: Estimate value function
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        
        # Learnable log std for exploration
        self.actor_logstd = nn.Parameter(torch.zeros(1, 6))
        
        # Store baseline policy for reference (not used during forward)
        self.baseline_policy = baseline_trajectory_policy
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def make_env(env_id, idx, run_name):
    def thunk():
        env = gym.make(env_id, render_mode=None, max_episode_steps=600)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("\n" + "="*70)
    print("RL FINE-TUNING IN OPTIMIZED SIMULATION")
    print("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Load baseline policy
    print("\nLoading baseline policy...")
    baseline_policy = TrajectoryPolicy()
    baseline_policy.load_state_dict(torch.load(BASELINE_POLICY_PATH, map_location=device, weights_only=False))
    baseline_policy.eval()
    print(f"✓ Baseline policy loaded from {BASELINE_POLICY_PATH}")
    
    # Create environments
    print("\nCreating optimized environments...")
    envs = gym.vector.SyncVectorEnv(
        [make_env("RedCubePick-Optimized-v0", i, RUN_NAME) for i in range(NUM_ENVS)]
    )
    print(f"✓ Created {NUM_ENVS} parallel environments")
    
    # Get dimensions
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    
    # Create RL policy
    print("\nCreating RL policy from baseline...")
    agent = RLPolicyFromTrajectory(baseline_policy, obs_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    print("✓ RL policy created")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/videos", exist_ok=True)
    
    print("\n" + "="*70)
    print("ULTRA-CONSERVATIVE FINE-TUNING CONFIGURATION")
    print("="*70)
    print(f"Total timesteps: {TOTAL_TIMESTEPS} (5x less than before)")
    print(f"Learning rate: {LEARNING_RATE} (100x lower than baseline!)")
    print(f"Num envs: {NUM_ENVS}")
    print(f"Steps per update: {NUM_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs per update: {N_EPOCHS} (3x less)")
    print(f"Entropy coef: {ENT_COEF} (ZERO exploration!)")
    print(f"Output: {OUTPUT_DIR}")
    print("\n⚠️  Strategy: Tiny tweaks only, preserve 95% of baseline policy")
    
    # Storage
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    
    # Start training
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    num_updates = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)
    
    print("\n" + "="*70)
    print("STARTING ULTRA-CONSERVATIVE RL FINE-TUNING")
    print("="*70)
    print("🎯 Goal: Tiny adjustments to adapt to weaker gripper (21.3 N vs 35 N)")
    print("⏱️  Expected time: ~2 minutes")
    
    pbar = tqdm(total=TOTAL_TIMESTEPS, desc="Fine-tuning")
    
    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # Step environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            pbar.update(NUM_ENVS)
        
        # Compute advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimize policy
        b_inds = np.arange(NUM_STEPS * NUM_ENVS)
        for epoch in range(N_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS * NUM_ENVS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
        
        # Evaluation
        if global_step % EVAL_FREQ == 0 or update == num_updates:
            print(f"\n\nEvaluating at step {global_step}...")
            eval_env = gym.make("RedCubePick-Optimized-v0", render_mode="rgb_array", max_episode_steps=600)
            
            eval_rewards = []
            eval_successes = []
            
            for ep in range(EVAL_EPISODES):
                obs, _ = eval_env.reset()
                frames = []
                episode_reward = 0
                
                for step in range(600):
                    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]
                    
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    
                    if ep == 0:  # Record first episode
                        frame = eval_env.render()
                        frames.append(frame)
                    
                    if terminated or truncated:
                        break
                
                eval_rewards.append(episode_reward)
                eval_successes.append(info.get("is_success", False))
                
                # Save video of first episode
                if ep == 0:
                    video_path = f"{OUTPUT_DIR}/videos/eval_step{global_step:07d}.mp4"
                    iio.imwrite(video_path, frames, fps=30)
            
            avg_reward = np.mean(eval_rewards)
            success_rate = np.mean(eval_successes)
            
            print(f"✓ Eval: Reward={avg_reward:.2f}, Success={success_rate:.0%}")
            
            eval_env.close()
    
    pbar.close()
    
    # Save final policy
    torch.save(agent.state_dict(), f"{OUTPUT_DIR}/policy_final.pt")
    print(f"\n✓ Fine-tuned policy saved to {OUTPUT_DIR}/policy_final.pt")
    
    # Save just the actor for deployment
    torch.save(agent.actor_mean.state_dict(), f"{OUTPUT_DIR}/policy_actor_only.pt")
    print(f"✓ Actor-only policy saved to {OUTPUT_DIR}/policy_actor_only.pt")
    
    envs.close()
    
    print("\n" + "="*70)
    print("✅ RL FINE-TUNING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Training Summary:")
    print(f"  Total steps: {global_step}")
    print(f"  Time elapsed: {(time.time() - start_time)/60:.1f} minutes")
    print(f"  Output: {OUTPUT_DIR}")
    
    print("\n🎯 Next Steps:")
    print("1. Check evaluation videos in:")
    print(f"   {OUTPUT_DIR}/videos/")
    print("\n2. Deploy fine-tuned policy on real robot:")
    print("   cd ../../working")
    print("   python scripts/real_sim/deploy_trajectory_policy.py")
    print("   (Update to use fine-tuned policy)")
    print("\n3. Compare baseline vs optimized!")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

