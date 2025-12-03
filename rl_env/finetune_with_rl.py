"""
RL fine-tuning with gripper penalty.

Use PPO to fine-tune the baseline policy in optimized sim with gripper penalty.
The agent will learn to grip LESS while still completing the task!
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
RUN_NAME = f"FINETUNED_RL__{int(time.time())}"
OUTPUT_DIR = f"runs/{RUN_NAME}"

# PPO Hyperparameters (ULTRA AGGRESSIVE to learn gripper reduction!)
LEARNING_RATE = 5e-5  # Even higher LR for faster learning!
NUM_ENVS = 4
NUM_STEPS = 512
TOTAL_TIMESTEPS = 50000  # More training to learn new behavior
BATCH_SIZE = 64
N_EPOCHS = 5
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.08  # Even more exploration to find lower gripper values!
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

EVAL_FREQ = 10000

# ============================================================================
# GRIPPER PENALTY WRAPPER
# ============================================================================

class GripperPenaltyWrapper(gym.Wrapper):
    """Penalizes excessive gripper closure to prevent overload."""
    
    def __init__(self, env, penalty_threshold=1.1, penalty_weight=100.0):
        super().__init__(env)
        self.penalty_threshold = penalty_threshold
        self.penalty_weight = penalty_weight
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get gripper position from observation
        gripper_pos = obs[5]
        
        # STRONG penalty for excessive closure (prevents overload!)
        if gripper_pos > self.penalty_threshold:
            excess = gripper_pos - self.penalty_threshold
            penalty = -self.penalty_weight * (excess ** 2)
            reward += penalty
            info["gripper_penalty"] = penalty
            info["gripper_pos"] = gripper_pos
        
        return obs, reward, terminated, truncated, info

# ============================================================================
# TIME-TRACKING WRAPPER
# ============================================================================

class TimeTrackingWrapper(gym.Wrapper):
    """Provides normalized time as observation."""
    
    def __init__(self, env, max_steps=200):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        
        # Override observation space to be just time
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
        time_obs = np.array([self.current_step / (self.max_steps - 1)], dtype=np.float32)
        return time_obs, reward, terminated, truncated, info

# ============================================================================
# RL POLICY
# ============================================================================

class TimeBased_RLPolicy(nn.Module):
    """RL policy initialized from baseline."""
    
    def __init__(self, baseline_policy):
        super().__init__()
        
        # Actor: Copy from baseline
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
        
        # Copy weights from baseline
        with torch.no_grad():
            self.actor.load_state_dict(baseline_policy.net.state_dict())
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        
        # Log std for exploration
        self.actor_logstd = nn.Parameter(torch.zeros(1, 6))
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        from torch.distributions.normal import Normal
        
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# ============================================================================
# CREATE ENVIRONMENT
# ============================================================================

def create_optimized_env(params):
    env = gym.make("RedCubePick-v0", render_mode=None, max_episode_steps=600)
    
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
    
    # Add gripper penalty (STRONG penalty!)
    env = GripperPenaltyWrapper(env, penalty_threshold=1.2, penalty_weight=50.0)
    
    # Add time tracking
    env = TimeTrackingWrapper(env, max_steps=200)
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    return env

def make_env(params, idx):
    def thunk():
        return create_optimized_env(params)
    return thunk

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("RL FINE-TUNING WITH GRIPPER PENALTY")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Load optimized parameters
    with open(OPTIMIZED_PARAMS_PATH, "r") as f:
        opt_data = json.load(f)
    opt_params = opt_data["optimized_parameters"]
    print("\n✓ Optimized parameters loaded")
    
    # Load baseline policy
    print("\nLoading baseline policy...")
    baseline_policy = TrajectoryPolicy()
    baseline_policy.load_state_dict(torch.load(BASELINE_POLICY_PATH, map_location=device, weights_only=False))
    baseline_policy.eval()
    print("✓ Baseline policy loaded")
    
    # Create environments
    print(f"\nCreating {NUM_ENVS} environments with gripper penalty...")
    envs = gym.vector.SyncVectorEnv(
        [make_env(opt_params, i) for i in range(NUM_ENVS)]
    )
    print("✓ Environments created")
    
    # Create RL policy
    print("\nCreating RL policy from baseline...")
    agent = TimeBased_RLPolicy(baseline_policy).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    print("✓ RL policy created")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/videos", exist_ok=True)
    
    print("\n" + "="*70)
    print("STARTING PPO FINE-TUNING")
    print("="*70)
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Gripper penalty: Penalize closure > 1.2 rad")
    print(f"Goal: Learn to grip LESS while still succeeding!")
    
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
    
    pbar = tqdm(total=TOTAL_TIMESTEPS, desc="RL Fine-tuning")
    
    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # Convert action (joint positions) to normalized action for env
            action_np = action.cpu().numpy()
            
            # Step environment
            next_obs, reward, terminations, truncations, infos = envs.step(action_np)
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
            
            # Create eval env with render mode
            eval_env_base = gym.make("RedCubePick-v0", render_mode="rgb_array", max_episode_steps=600)
            
            # Apply optimized parameters
            for i in range(eval_env_base.unwrapped.model.nu):
                actuator_name = eval_env_base.unwrapped.model.actuator(i).name
                if "Jaw" in actuator_name:
                    eval_env_base.unwrapped.model.actuator_gainprm[i, 0] = opt_params["gripper_kp"]
                    eval_env_base.unwrapped.model.actuator_biasprm[i, 1] = -opt_params["gripper_kp"]
                    eval_env_base.unwrapped.model.actuator_forcerange[i, 0] = -opt_params["gripper_forcerange"]
                    eval_env_base.unwrapped.model.actuator_forcerange[i, 1] = opt_params["gripper_forcerange"]
                else:
                    eval_env_base.unwrapped.model.actuator_gainprm[i, 0] = opt_params["arm_kp"]
                    eval_env_base.unwrapped.model.actuator_biasprm[i, 1] = -opt_params["arm_kp"]
            
            for i in range(eval_env_base.unwrapped.model.nv):
                eval_env_base.unwrapped.model.dof_damping[i] = opt_params["joint_damping"]
            
            for i in range(eval_env_base.unwrapped.model.ngeom):
                geom_name = eval_env_base.unwrapped.model.geom(i).name
                if "jaw_pad" in geom_name.lower():
                    eval_env_base.unwrapped.model.geom_friction[i, 0] = opt_params["gripper_friction"]
            
            cube_body_id = eval_env_base.unwrapped.cube_body_id
            eval_env_base.unwrapped.model.body_mass[cube_body_id] = opt_params["cube_mass"]
            
            obs, _ = eval_env_base.reset()
            frames = []
            episode_reward = 0
            gripper_values = []
            
            for step in range(200):
                # Get action from policy (time-based)
                time_norm = step / 199.0
                time_tensor = torch.FloatTensor([[time_norm]]).to(device)
                
                with torch.no_grad():
                    target_joints = agent.actor(time_tensor).cpu().numpy()[0]
                
                gripper_values.append(target_joints[5])
                
                # Convert to normalized action
                ctrl_low = eval_env_base.unwrapped.ctrl_low
                ctrl_high = eval_env_base.unwrapped.ctrl_high
                
                action = np.zeros(6, dtype=np.float32)
                action[:5] = 2.0 * (target_joints[:5] - ctrl_low[:5]) / (ctrl_high[:5] - ctrl_low[:5]) - 1.0
                action[:5] = np.clip(action[:5], -1.0, 1.0)
                action[5] = 2.0 * (target_joints[5] - ctrl_low[5]) / (ctrl_high[5] - ctrl_low[5]) - 1.0
                action[5] = np.clip(action[5], -1.0, 1.0)
                
                obs, reward, terminated, truncated, info = eval_env_base.step(action)
                episode_reward += reward
                frames.append(eval_env_base.render())
            
            eval_env_base.close()
            
            # Save video
            video_path = f"{OUTPUT_DIR}/videos/eval_step{global_step:07d}.mp4"
            iio.imwrite(video_path, frames, fps=30)
            
            max_gripper = max(gripper_values)
            print(f"✓ Video saved: {video_path}")
            print(f"✓ Reward: {episode_reward:.2f}")
            print(f"✓ Max gripper: {max_gripper:.4f} rad (baseline was 1.7416)")
            
            reduction = (1.7416 - max_gripper) / 1.7416 * 100
            if max_gripper < 1.5:
                print(f"🎉 GRIPPER REDUCED BY {reduction:.1f}%! Now closing to {max_gripper:.4f} rad!")
            else:
                print(f"⚠️  Still closing to {max_gripper:.4f} rad (need < 1.5)")
    
    pbar.close()
    
    # Save final policy (just the actor)
    torch.save(agent.actor.state_dict(), f"{OUTPUT_DIR}/policy_finetuned.pt")
    print(f"\n✓ Fine-tuned policy saved to {OUTPUT_DIR}/policy_finetuned.pt")
    
    envs.close()
    
    print("\n" + "="*70)
    print("✅ RL FINE-TUNING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Training Summary:")
    print(f"  Total steps: {global_step}")
    print(f"  Time elapsed: {(time.time() - start_time)/60:.1f} minutes")
    print(f"  Output: {OUTPUT_DIR}")
    
    print("\n🎯 Next: Deploy on real robot!")

if __name__ == "__main__":
    main()

