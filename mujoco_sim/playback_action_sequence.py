#!/usr/bin/env python3
"""
Script to replay sparse waypoints from teleop.json, interpolate them smoothly in MuJoCo,
and save a denser trajectory (sampled every 0.5s) to teleop_dense.json.
"""

import json
import time
import math
from pathlib import Path
from typing import List

import mujoco
import numpy as np

# Load configuration
TELEOP_PATH = Path("mujoco_sim/teleop.json")
OUTPUT_PATH = Path("mujoco_sim/teleop_dense.json")
SCENE_PATH = Path("mujoco_sim/trs_so_arm100/scene.xml")

JOINT_SPEED = 1.5  # rad/s
MIN_MOVE_TIME = 0.2  # seconds
GRIPPER_TIME = 0.5   # seconds
SAMPLE_RATE = 0.5    # seconds (record data every 0.5s)

def load_sequences(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def main():
    if not TELEOP_PATH.exists():
        print(f"Error: {TELEOP_PATH} not found.")
        return

    print(f"Loading sequences from {TELEOP_PATH}...")
    data = load_sequences(TELEOP_PATH)
    
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    
    # Get home keyframe ID
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_cube")
    
    dense_data = {}
    
    # Iterate over each sequence in the input file
    for seq_key, sequence in sorted(data.items()):
        if not seq_key.startswith("ACTION_SEQUENCE"):
            continue
            
        print(f"Processing {seq_key} ({len(sequence)} waypoints)...")
        
        # Reset simulation for each sequence
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data_mj, home_key_id)
            mujoco.mj_forward(model, data_mj)
        
        # Start recording buffer
        recorded_trajectory = []
        
        # Initial state
        current_qpos = data_mj.qpos[:6].copy()
        data_mj.ctrl[:6] = current_qpos
        
        # We simulate time progression manually
        sim_time = 0.0
        last_record_time = -SAMPLE_RATE # Force record at t=0
        
        # Move through waypoints
        # Assume robot starts at first waypoint if needed, or just interpolate from home
        # The teleop.json sequences usually start with Home
        
        for i in range(len(sequence) - 1):
            start_pose = np.array(sequence[i], dtype=np.float64)
            end_pose = np.array(sequence[i+1], dtype=np.float64)
            
            # Determine duration
            # Check if it's a gripper action (joints same, gripper changes)
            joint_dist = np.linalg.norm(start_pose[:5] - end_pose[:5])
            gripper_dist = abs(start_pose[-1] - end_pose[-1])
            
            if joint_dist < 0.01 and gripper_dist > 0.1:
                duration = GRIPPER_TIME
            else:
                duration = max(joint_dist / JOINT_SPEED, MIN_MOVE_TIME)
            
            # Simulate this segment
            # We step by physics timestep
            dt = model.opt.timestep
            steps = int(math.ceil(duration / dt))
            
            for s in range(steps):
                # Interpolate
                t = (s + 1) / steps
                # Smoothstep for nicer motion: 3t^2 - 2t^3
                smooth_t = t * t * (3 - 2 * t)
                
                target_cmd = start_pose + (end_pose - start_pose) * smooth_t
                
                # Apply control
                data_mj.ctrl[:6] = target_cmd
                
                # Step physics
                mujoco.mj_step(model, data_mj)
                sim_time += dt
                
                # Record if enough time has passed
                if sim_time - last_record_time >= SAMPLE_RATE:
                    # Record current ACTUAL position (qpos), not command
                    recorded_trajectory.append(data_mj.qpos[:6].tolist())
                    last_record_time = sim_time
            
            # Ensure we reach the exact end pose of this segment in our internal state logic
            # (Physics might lag slightly, but for planning next segment start, use theoretical)
            # Actually, better to use end_pose as start for next segment to avoid drift in command
            pass

        # Ensure final pose is recorded if we haven't just recorded it
        if sim_time - last_record_time > 0.01:
             recorded_trajectory.append(data_mj.qpos[:6].tolist())

        dense_data[seq_key] = recorded_trajectory
        print(f"  -> Recorded {len(recorded_trajectory)} frames")

    # Save output
    print(f"Saving dense trajectories to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(dense_data, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    main()
