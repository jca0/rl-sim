#!/usr/bin/env python3
"""Minimal CleanRL-style PPO trainer with video/joint logging."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

import rl_env  # noqa: F401  # registers the env


ENV_ID = "RedCubePick-v0"
VIDEO_FPS = 30
EVAL_MAX_STEPS = 400

# Fixed PPO hyperparameters (kept out of CLI to stay simple).
NUM_ENVS = 8
NUM_STEPS = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 10
NUM_MINIBATCHES = 32
LEARNING_RATE = 3e-4
CLIP_COEF = 0.2
ENT_COEF = 0.01  # Non-zero for exploration!
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CLIP_VALUE_LOSS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train RedCubePick PPO with logging artifacts.")
    parser.add_argument("--exp-name", type=str, default="red_cube_pick")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--video-interval", type=int, default=100_000,
                        help="Steps between saved rollout videos (0 disables).")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--save-policy", action="store_true", help="Persist final weights.")
    return parser.parse_args()


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-critic network modeled after CleanRL's PPO continuous agent."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd)
        probs = Normal(mean, std)
        if action is None:
            raw_action = probs.sample()
        else:
            raw_action = action

        logprob = probs.log_prob(raw_action).sum(-1)
        entropy = probs.entropy().sum(-1)
        squashed = torch.tanh(raw_action)
        return squashed, logprob, entropy, self.critic(x), raw_action

    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.actor_mean(x)
        return torch.tanh(mean)


@dataclass
class RolloutArtifact:
    video_path: Path
    joints_path: Path
    episode_return: float
    episode_length: int


def record_rollout(
    agent: Agent,
    device: torch.device,
    global_step: int,
    video_dir: Path,
    joint_dir: Path,
    seed: int,
) -> RolloutArtifact:
    video_dir.mkdir(parents=True, exist_ok=True)
    joint_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    frames: List[np.ndarray] = []
    true_env = env.unwrapped
    joints_log: List[Dict[str, Any]] = [
        {"time_s": 0.0, "qpos": true_env.current_joint_positions().tolist()}
    ]
    elapsed = 0.0
    next_sample = 1.0
    total_reward = 0.0

    for _ in range(EVAL_MAX_STEPS):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = agent.act_deterministic(obs_tensor).squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        elapsed += true_env.dt

        while elapsed >= next_sample:
            joints_log.append(
                {
                    "time_s": round(float(next_sample), 4),
                    "qpos": true_env.current_joint_positions().tolist(),
                }
            )
            next_sample += 1.0

        if terminated or truncated:
            break

    env.close()

    if not frames:
        raise RuntimeError("No frames captured during rollout; cannot encode video.")

    video_path = video_dir / f"step_{global_step:08d}.mp4"
    iio.imwrite(video_path, np.asarray(frames), fps=VIDEO_FPS, codec="libx264", quality=8)

    joints_path = joint_dir / f"step_{global_step:08d}.json"
    with joints_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "global_step": global_step,
                "episode_return": total_reward,
                "episode_length": len(frames),
                "joints": joints_log,
            },
            f,
            indent=2,
        )

    return RolloutArtifact(video_path, joints_path, total_reward, len(frames))


def make_env(seed: int, idx: int) -> gym.Env:
    def thunk() -> gym.Env:
        env = gym.make(ENV_ID)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    run_dir = Path(args.output_dir).expanduser() / run_name
    video_dir = run_dir / "videos"
    joint_dir = run_dir / "joint_logs"
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(run_dir.as_posix())
    writer.add_text(
        "config",
        json.dumps(
            {
                "seed": args.seed,
                "total_timesteps": args.total_timesteps,
                "video_interval": args.video_interval,
                "device": str(device),
            },
            indent=2,
        ),
    )

    envs = gym.vector.SyncVectorEnv([make_env(args.seed, i) for i in range(NUM_ENVS)])
    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape

    agent = Agent(envs.single_observation_space, envs.single_action_space).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + obs_shape, device=device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + act_shape, device=device)
    logprobs = torch.zeros(NUM_STEPS, NUM_ENVS, device=device)
    rewards = torch.zeros(NUM_STEPS, NUM_ENVS, device=device)
    dones = torch.zeros(NUM_STEPS, NUM_ENVS, device=device)
    values = torch.zeros(NUM_STEPS, NUM_ENVS, device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(NUM_ENVS, device=device)

    global_step = 0
    next_video_step = args.video_interval if args.video_interval > 0 else None
    num_updates = args.total_timesteps // (NUM_STEPS * NUM_ENVS)
    start_time = time.time()

    progress = trange(1, num_updates + 1, desc="PPO updates", dynamic_ncols=True)
    sps_value = 0
    last_loss = 0.0

    for update in progress:
        for step in range(NUM_STEPS):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                env_action, logprob, entropy, value, raw_action = agent.get_action_and_value(next_obs)

            actions[step] = raw_action
            logprobs[step] = logprob
            values[step] = value.squeeze(-1)

            env_action_np = env_action.cpu().numpy()
            next_obs_np, reward, terminated, truncated, info = envs.step(env_action_np)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(np.logical_or(terminated, truncated), dtype=torch.float32, device=device)

            global_step += NUM_ENVS

            if "final_info" in info:
                for episode_info in info["final_info"]:
                    if episode_info is not None:
                        writer.add_scalar("charts/episode_return", episode_info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episode_length", episode_info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)

        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = torch.zeros(NUM_ENVS, device=device)
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = NUM_ENVS * NUM_STEPS
        minibatch_size = batch_size // NUM_MINIBATCHES
        clipfracs: List[float] = []
        approx_kl_vals: List[float] = []

        for epoch in range(UPDATE_EPOCHS):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = indices[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > CLIP_COEF).float().mean().item())
                    approx_kl_vals.append(((ratio - 1) - logratio).mean().item())

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                if CLIP_VALUE_LOSS:
                    value_pred_clipped = mb_values + (newvalue - mb_values).clamp(-CLIP_COEF, CLIP_COEF)
                    value_losses = (newvalue - mb_returns) ** 2
                    value_losses_clipped = (value_pred_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = policy_loss - ENT_COEF * entropy_loss + VF_COEF * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        approx_kl = float(np.mean(approx_kl_vals)) if approx_kl_vals else 0.0
        clipfrac = float(np.mean(clipfracs)) if clipfracs else 0.0

        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", clipfrac, global_step)
        sps_value = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps_value, global_step)

        if next_video_step is not None and global_step >= next_video_step:
            artifact = record_rollout(
                agent=agent,
                device=device,
                global_step=global_step,
                video_dir=video_dir,
                joint_dir=joint_dir,
                seed=args.seed,
            )
            writer.add_scalar("eval/episode_return", artifact.episode_return, global_step)
            print(f"[video] step={global_step} -> {artifact.video_path}")
            next_video_step += args.video_interval

        last_loss = loss.item()
        progress.set_postfix({"loss": f"{last_loss:.3f}", "SPS": sps_value})

    envs.close()
    writer.close()

    if args.save_policy:
        torch.save(agent.state_dict(), run_dir / "policy.pt")


if __name__ == "__main__":
    main()
