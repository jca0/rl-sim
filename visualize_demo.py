#!/usr/bin/env python3
"""Visualize a random demonstration sequence as a video to verify data quality."""

import json
import random
import math
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

# Config
SCENE_PATH = Path("mujoco_sim/trs_so_arm100/scene.xml")
TELEOP_PATH = Path("mujoco_sim/teleop.json")
OUTPUT_VIDEO = Path("demo_visualization.mp4")

JOINT_SPEED = 1.5  # rad/s
MIN_MOVE_TIME = 0.2
GRIPPER_TIME = 0.5
VIDEO_FPS = 30


def main():
    print("=" * 60)
    print("DEMO VISUALIZATION - Verifying Teleop Data")
    print("=" * 60)
    
    # Load demos
    with open(TELEOP_PATH) as f:
        data = json.load(f)
    
    sequences = {k: v for k, v in data.items() if k.startswith("ACTION_SEQUENCE")}
    print(f"\nFound {len(sequences)} demonstration sequences")
    
    # Pick a random sequence
    seq_key = random.choice(list(sequences.keys()))
    sequence = sequences[seq_key]
    
    print(f"\nVisualizing: {seq_key}")
    print(f"Waypoints: {len(sequence)}")
    for i, waypoint in enumerate(sequence):
        print(f"  Waypoint {i}: {waypoint}")
    
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Reset to home
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_cube")
    mujoco.mj_resetDataKeyframe(model, data_mj, home_key_id)
    mujoco.mj_forward(model, data_mj)
    
    frames = []
    
    # Render initial state
    renderer.update_scene(data_mj)
    frames.append(renderer.render().copy())
    
    print(f"\nSimulating motion...")
    
    # Execute each segment
    for i in range(len(sequence) - 1):
        start_pose = np.array(sequence[i], dtype=np.float64)
        end_pose = np.array(sequence[i+1], dtype=np.float64)
        
        # Determine duration
        joint_dist = np.linalg.norm(start_pose[:5] - end_pose[:5])
        gripper_dist = abs(start_pose[-1] - end_pose[-1])
        
        if joint_dist < 0.01 and gripper_dist > 0.1:
            duration = GRIPPER_TIME
            move_type = "GRASP" if end_pose[-1] > 0.5 else "RELEASE"
        else:
            duration = max(joint_dist / JOINT_SPEED, MIN_MOVE_TIME)
            move_type = "MOVE"
        
        print(f"  Segment {i} -> {i+1}: {move_type}, duration={duration:.2f}s")
        
        # Simulate segment
        dt = model.opt.timestep
        steps = int(math.ceil(duration / dt))
        
        for s in range(steps):
            t = (s + 1) / steps
            # Smooth interpolation
            smooth_t = t * t * (3 - 2 * t)
            target_cmd = start_pose + (end_pose - start_pose) * smooth_t
            
            # Apply control
            data_mj.ctrl[:6] = target_cmd
            
            # Step physics
            mujoco.mj_step(model, data_mj)
            
            # Render every few steps to match video FPS
            if s % 5 == 0 or s == steps - 1:
                renderer.update_scene(data_mj)
                frames.append(renderer.render().copy())
    
    print(f"\nCaptured {len(frames)} frames")
    
    # Save video
    print(f"Saving video to {OUTPUT_VIDEO}...")
    iio.imwrite(OUTPUT_VIDEO, np.array(frames), fps=VIDEO_FPS, codec="libx264", quality=8)
    
    print(f"\n✅ Done! Watch {OUTPUT_VIDEO} to verify the demo is correct.")
    print("=" * 60)
    
    # Cleanup
    if hasattr(renderer, 'close'):
        renderer.close()


if __name__ == "__main__":
    main()

