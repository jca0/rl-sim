"""
PPO fine-tuning with TIME-BASED policy in optimized simulation.

Key Insight: The baseline policy is time → joints. We keep this structure!
We just let PPO adjust the joint positions slightly to compensate for the
weaker gripper force in the optimized sim.

Architecture: TrajectoryPolicy (time → joints) + value head
Training: PPO with baseline policy as initialization
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
RUN_NAME = f"FINETUNED_PPO_CORRECT__{int(time.time())}"
OUTPUT_DIR = f"runs/{RUN_NAME}"

# PPO Hyperparameters (conservative)
LEARNING_RATE = 3e-5  # Moderate LR
NUM_ENVS = 4
NUM_STEPS = 512  # Shorter rollouts
TOTAL_TIMESTEPS = 50000
BATCH_SIZE = 64
N_EPOCHS = 5  # Fewer epochs
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.02  # Small exploration
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

EVAL_FREQ = 10000

# ============================================================================
# TIME-BASED RL POLICY
# ============================================================================

class TimeBased_RLPolicy(nn.Module):
    """
    RL policy that uses TIME as input (like baseline), not full observation.
    
    Actor: time → joint positions (initialized from baseline)
    Critic: time → value
    """
    
    def __init__(self, baseline_policy):
        super().__init__()
        
        # Actor: EXACT same architecture as baseline (1→512→1024→1024→512→256→6)
        self.actor = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        
        # Copy ALL weights from baseline policy (exact architecture match!)
        with torch.no_grad():
            self.actor.load_state_dict(baseline_policy.net.state_dict())
        
        # Critic: Estimate value from time
        self.critic = nn.Sequential(
            nn.Linear(1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        
        # Learnable log std for exploration
        self.actor_logstd = nn.Parameter(torch.zeros(1, 6))
    
    def get_value(self, time_input):
        return self.critic(time_input)
    
    def get_action_and_value(self, time_input, action=None):
        from torch.distributions.normal import Normal
        
        action_mean = self.actor(time_input)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(time_input)

# ============================================================================
# TIME-TRACKING WRAPPER
# ============================================================================

class TimeTrackingWrapper(gym.Wrapper):
    """
    Wrapper that tracks episode time and provides it as observation.
    """
    
    def __init__(self, env, max_steps=200):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        
        # Override observation space to be just time (1D)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        return np.array([0.0], dtype=np.float32), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Return normalized time as observation
        time_obs = np.array([self.current_step / (self.max_steps - 1)], dtype=np.float32)
        
        return time_obs, reward, terminated, truncated, info

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def make_env(env_id, idx, run_name):
    def thunk():
        env = gym.make(env_id, render_mode=None, max_episode_steps=600)
        env = TimeTrackingWrapper(env, max_steps=200)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PPO FINE-TUNING WITH TIME-BASED POLICY")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Load baseline policy
    print("\nLoading baseline policy...")
    baseline_policy = TrajectoryPolicy()
    baseline_policy.load_state_dict(torch.load(BASELINE_POLICY_PATH, map_location=device, weights_only=False))
    baseline_policy.eval()
    print(f"✓ Baseline policy loaded")
    
    # Create environments
    print("\nCreating optimized environments...")
    envs = gym.vector.SyncVectorEnv(
        [make_env("RedCubePick-Optimized-v0", i, RUN_NAME) for i in range(NUM_ENVS)]
    )
    print(f"✓ Created {NUM_ENVS} parallel environments")
    
    # Create RL policy (initialized from baseline)
    print("\nCreating RL policy from baseline...")
    agent = TimeBased_RLPolicy(baseline_policy).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    print("✓ RL policy created and initialized from baseline")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/videos", exist_ok=True)
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Num envs: {NUM_ENVS}")
    print(f"Entropy coef: {ENT_COEF}")
    print(f"Input: TIME (normalized 0-1)")
    print(f"Output: 6 joint positions")
    print(f"Initialization: Baseline policy weights")
    
    # Storage
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, 1)).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS, 6)).to(device)
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
    print("STARTING PPO FINE-TUNING")
    print("="*70)
    
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
        b_obs = obs.reshape((-1, 1))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 6))
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
            
            obs, _ = eval_env.reset()
            frames = []
            episode_reward = 0
            
            for step in range(200):
                t_norm = step / 199.0
                t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(t_tensor)
                action = action.cpu().numpy()[0]
                
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                frame = eval_env.render()
                frames.append(frame)
                
                if terminated or truncated:
                    break
            
            video_path = f"{OUTPUT_DIR}/videos/eval_step{global_step:07d}.mp4"
            iio.imwrite(video_path, frames, fps=30)
            print(f"✓ Video saved: {video_path}")
            print(f"✓ Episode reward: {episode_reward:.2f}")
            
            eval_env.close()
    
    pbar.close()
    
    # Save final policy (just the actor)
    torch.save(agent.actor.state_dict(), f"{OUTPUT_DIR}/policy_actor.pt")
    print(f"\n✓ Fine-tuned policy saved to {OUTPUT_DIR}/policy_actor.pt")
    
    envs.close()
    
    print("\n" + "="*70)
    print("✅ PPO FINE-TUNING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Training Summary:")
    print(f"  Total steps: {global_step}")
    print(f"  Time elapsed: {(time.time() - start_time)/60:.1f} minutes")
    print(f"  Output: {OUTPUT_DIR}")
    
    print("\n🎯 Next Steps:")
    print("1. Check evaluation videos")
    print("2. Deploy on real robot if videos look good!")

if __name__ == "__main__":
    main()

