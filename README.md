# rl-sim
**running sim RL script:**  
example:  
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m rl_env.train_cube_robot_env.py \
    --total-timesteps 2_000_000 \
    --video-interval 200_000 \
    --save-policy
```
saves the outputs to `runs/` folder

export PYTHONPATH=$PYTHONPATH:$(pwd)