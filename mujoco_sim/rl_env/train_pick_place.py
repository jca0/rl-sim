"""
CleanRL-style PPO trainer for the SO-100 pick-and-place environment.

Based on https://github.com/vwxyzjn/cleanrl (MIT License). The script keeps the
same algorithmic structure while swapping in the custom MuJoCo environment.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import ClipAction, RecordEpisodeStatistics, RecordVideo
from torch import nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

if __package__:
    from .pick_place_env import SOPickPlaceEnv
else:
    from pick_place_env import SOPickPlaceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO training for SO-100 pick & place")
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__)[:-3],
        help="experiment name",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", action="store_true", default=False)
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Enable CUDA if available"
    )
    parser.add_argument("--track", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project-name", type=str, default="so100-pick-place")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", action="store_true")

    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument(
        "--anneal-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly anneal the learning rate over updates.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Where to store the final policy (defaults to runs/<run_name>/policy.pt)",
    )
    parser.add_argument(
        "--action-seq-length",
        type=int,
        default=200,
        help="Maximum number of steps to export into a joint-position sequence.",
    )
    parser.add_argument(
        "--action-seq-path",
        type=str,
        default=None,
        help="Optional path for exported ACTION_SEQUENCE (defaults to runs/<run_name>/action_sequence.py).",
    )
    return parser.parse_args()


def make_env(
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = SOPickPlaceEnv(seed=seed + idx, render_mode=render_mode)
        env = RecordEpisodeStatistics(env, deque_size=20)
        if capture_video and idx == 0:
            video_dir = os.path.join("runs", run_name, "videos")
            os.makedirs(video_dir, exist_ok=True)

            def video_trigger(step: int) -> bool:
                return step % 5000 == 0

            env = RecordVideo(env, video_dir, step_trigger=video_trigger, video_length=400)
        env = ClipAction(env)
        return env

    return thunk


class Agent(nn.Module):
    def __init__(self, obs_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()
        obs_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(action_space.shape))

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(256, 1)

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.network(x)
        mean = self.actor_mean(hidden)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(hidden)
        return action, log_prob, entropy, value


def evaluate_policy(
    agent: Agent, device: torch.device, episodes: int, seed: int
) -> tuple[float, float]:
    env = SOPickPlaceEnv(seed=seed + 10_000)
    returns: List[float] = []
    successes: List[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        ep_return = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    obs_tensor, deterministic=True
                )
            clipped = np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(clipped)
            ep_return += reward
            done = terminated or truncated
            if done:
                returns.append(ep_return)
                successes.append(info.get("success", 0.0))
    env.close()
    return float(np.mean(returns)), float(np.mean(successes))


def export_action_sequence(
    agent: Agent,
    device: torch.device,
    seed: int,
    horizon: int,
    output_path: str,
) -> List[tuple[float, ...]]:
    env = SOPickPlaceEnv(seed=seed + 20_000)
    obs, _ = env.reset(seed=seed + 20_000)
    sequence: List[tuple[float, ...]] = []

    for _ in range(horizon):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                obs_tensor, deterministic=True
            )
        clipped = np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)
        obs, _, terminated, truncated, _ = env.step(clipped)
        sequence.append(tuple(float(x) for x in env.ctrl_target))
        if terminated or truncated:
            break
    env.close()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated joint targets from SOPickPlaceEnv PPO policy\n")
        f.write("ACTION_SEQUENCE = [\n")
        for joints in sequence:
            joint_str = ", ".join(f"{value:.6f}" for value in joints)
            f.write(f"    ({joint_str}),\n")
        f.write("]\n")

    return sequence


def main() -> None:
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb_run = None
    if args.track:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "--track requires the 'wandb' package. Install it or omit --track."
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = SyncVectorEnv(
        [
            make_env(args.seed, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box)
    agent = Agent(envs.single_observation_space, envs.single_action_space).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    writer = SummaryWriter(os.path.join("runs", run_name))
    global_step = 0
    start_time = time.time()

    obs_shape = (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    act_shape = (args.num_steps, args.num_envs) + envs.single_action_space.shape

    obs = torch.zeros(obs_shape).to(device)
    actions = torch.zeros(act_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    rollout_size = args.num_envs * args.num_steps
    if args.total_timesteps % rollout_size != 0:
        raise ValueError(
            "total_timesteps must be divisible by num_envs * num_steps for PPO."
        )
    num_updates = args.total_timesteps // rollout_size

    progress_bar = tqdm(range(1, num_updates + 1), desc="Training", ncols=100)
    for update in progress_bar:
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        for step in range(args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done
            global_step += args.num_envs

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()

            clipped_actions = np.clip(action.cpu().numpy(), -1.0, 1.0)
            next_obs_np, reward, terminated, truncated, infos = envs.step(clipped_actions)
            rewards[step] = torch.tensor(reward).to(device)
            next_done = torch.tensor(
                terminated | truncated, dtype=torch.float32, device=device
            )

            for info in infos:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    if "success" in info:
                        writer.add_scalar("charts/success", info["success"], global_step)

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = torch.zeros(args.num_envs).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
                advantages[t] = lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = rollout_size
        if batch_size % args.num_minibatches != 0:
            raise ValueError("num_minibatches must divide num_envs * num_steps.")
        minibatch_size = batch_size // args.num_minibatches
        inds = np.arange(batch_size)
        clipfracs: List[float] = []
        approx_kl = torch.tensor(0.0)

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )
                    approx_kl = ((ratio - 1.0) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        explained_var = 1 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        progress_bar.set_postfix({"SPS": sps, "updates": update})

    progress_bar.close()

    eval_return, eval_success = evaluate_policy(
        agent, device, args.eval_episodes, args.seed
    )
    writer.add_scalar("eval/return", eval_return, global_step)
    writer.add_scalar("eval/success", eval_success, global_step)
    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/return": eval_return,
                "eval/success": eval_success,
            },
            step=global_step,
        )

    save_path = args.checkpoint_path or os.path.join("runs", run_name, "policy.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(agent.state_dict(), save_path)

    action_seq_path = args.action_seq_path or os.path.join(
        "runs", run_name, "action_sequence.py"
    )
    sequence = export_action_sequence(
        agent,
        device,
        args.seed,
        args.action_seq_length,
        action_seq_path,
    )
    writer.add_text(
        "artifacts/action_sequence_path",
        action_seq_path,
        global_step,
    )
    if wandb_run is not None:
        wandb_run.log({"artifacts/action_sequence_len": len(sequence)}, step=global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

