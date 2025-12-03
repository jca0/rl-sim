# rl_env: Training Scripts and Environment

## Files

| File | Purpose |
|------|---------|
| `cube_robot_env.py` | Main Gymnasium environment `RedCubePick-v0` |
| `train_trajectory_tracker.py` | Baseline policy training (time → joint positions) |
| `finetune_with_rl.py` | PPO fine-tuning with gripper penalty |
| `__init__.py` | Registers environment with Gymnasium |

## Environment: RedCubePick-v0

**Observation (12D):** 6 joint positions + 6 joint velocities  
**Action (6D):** Normalized joint commands [-1, 1]  
**Reward:** Reach + Grasp + Place (gated)

```python
import gymnasium as gym
import rl_env  # Registers the environment

env = gym.make("RedCubePick-v0", render_mode="rgb_array")
obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Training Methods

### 1. Trajectory Tracker (Baseline)
Supervised learning: time → joint positions
```bash
python -m rl_env.train_trajectory_tracker
```
- Uses 50 human demos from `teleop.json`
- 2000 epochs, ~10 minutes on RTX 3070 Ti
- Output: `runs/FINAL_FIXED__*/policy_epoch2000.pt`

### 2. RL Fine-tuning (Gripper Penalty)
PPO with gripper penalty to reduce closure force
```bash
python -m rl_env.finetune_with_rl
```
- Loads baseline policy
- Adds penalty for gripper > 1.1 rad
- 50k timesteps, ~1 minute
- Output: `runs/FINETUNED_RL__*/policy_finetuned.pt`

## failed_training/
Contains experimental training scripts that did not produce working policies (BC, ACT, LSTM, PPO variants). Kept for reference.

