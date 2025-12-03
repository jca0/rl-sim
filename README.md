# RL-Sim: MuJoCo Simulation for SO-100 Robot

Reinforcement learning environment for training pick-and-place policies in MuJoCo simulation.

## Directory Structure

```
rl-sim/
├── rl_env/                    # Training scripts and environment
│   ├── cube_robot_env.py      # Main Gymnasium environment (RedCubePick-v0)
│   ├── train_trajectory_tracker.py  # Baseline policy training
│   └── finetune_with_rl.py    # RL fine-tuning with gripper penalty
├── mujoco_sim/                # MuJoCo scene and teleoperation
│   ├── trs_so_arm100/         # Robot URDF/XML and assets
│   ├── teleop.json            # 50 human demonstrations
│   └── keyboard_teleop.py     # Manual control
├── runs/                      # Training outputs
│   ├── FINAL_FIXED__*/        # Baseline policy (working)
│   └── FINETUNED_RL__*/       # Fine-tuned policy (reduced gripper)
├── visualize_demo.py          # Visualize single demo
└── visualize_demo_grid.py     # Visualize demos in grid
```

## Quick Start

### Train Baseline Policy
```bash
cd submodule/rl-sim
python -m rl_env.train_trajectory_tracker
```

### Fine-tune with Gripper Penalty
```bash
python -m rl_env.finetune_with_rl
```

### Visualize Demos
```bash
python visualize_demo.py
python visualize_demo_grid.py --teleop_path mujoco_sim/teleop.json
```

## Key Files

| File | Purpose |
|------|---------|
| `rl_env/cube_robot_env.py` | Gymnasium environment with reward shaping |
| `rl_env/train_trajectory_tracker.py` | Time→joints supervised learning |
| `rl_env/finetune_with_rl.py` | PPO fine-tuning with gripper penalty |
| `mujoco_sim/teleop.json` | 50 human demonstrations (working) |

## Training Results

- **Baseline:** `runs/FINAL_FIXED__1764557628/policy_epoch2000.pt`
- **Fine-tuned:** `runs/FINETUNED_RL__1764660601/policy_finetuned.pt`
- Gripper reduction: 1.74 → 1.43 rad (18% decrease)
