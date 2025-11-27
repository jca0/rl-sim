# SO-100 Pick & Place RL

This directory contains a CleanRL-style PPO trainer plus a custom Gymnasium
environment for the SO-100 MuJoCo scene (`trs_so_arm100/scene.xml`). The agent
observes joint states, cube pose, the 10 cm-left placement target, and several
task-specific vectors, and it outputs normalized joint-position deltas that are
smoothed into MuJoCo's position actuators.

## Environment

- Defined in `pick_place_env.py`
- Resets to the `rest_with_cube` keyframe and randomizes the cube ±1 cm on the
  table plus target jitter.
- Reward shaping encourages: reaching, lifting, closing the gripper, moving the
  cube 10 cm left, and staying near the goal. Success is reaching the target
  while lifted above `lift_height`.

## Training with PPO

Install dependencies (see `requirements.txt`) and then run:

```bash
python mujoco_sim/rl_env/train_pick_place.py \
    --total-timesteps 200000 \
    --num-envs 4 \
    --cuda \
    --capture-video
```

Key artifacts:

- TensorBoard logs under `runs/<run_name>/`
- Optional videos when `--capture-video` is set
- Trained weights at `runs/<run_name>/policy.pt` (custom path via
  `--checkpoint-path`)
- Auto-generated joint targets at `runs/<run_name>/action_sequence.py`
  (format matches `mujoco_sim/mujoco_pick_place.py`’s `ACTION_SEQUENCE` and can
  be copied to the real-robot script)

Use `python mujoco_sim/rl_env/train_pick_place.py --help` for the full list of
hyper-parameters (annealing, KL early-stop, evaluation episodes, W&B logging,
etc.).

