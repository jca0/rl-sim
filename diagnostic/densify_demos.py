#!/usr/bin/env python3
"""
Densify demonstration data by simulating the demos and recording every state-action pair.

This creates a TRULY dense dataset that policies can actually learn from.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = Path("mujoco_sim/trs_so_arm100/scene.xml")
INPUT_PATH = Path("mujoco_sim/teleop.json")
OUTPUT_PATH = Path("mujoco_sim/teleop_WORKING_dense.json")

JOINT_SPEED = 1.5  # rad/s
MIN_MOVE_TIME = 0.2
GRIPPER_TIME = 0.5

def densify_demos():
    print("=" * 60)
    print("DENSIFYING DEMOS")
    print("=" * 60)
    
    # Load sparse demos
    with open(INPUT_PATH) as f:
        sparse_data = json.load(f)
    
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    
    # Boost gripper
    for i in range(model.nu):
        if "Jaw" in model.actuator(i).name:
            model.actuator_gainprm[i, 0] = 500.0
            model.actuator_biasprm[i, 1] = -500.0
    
    dense_data = {}
    
    for seq_key in sorted(sparse_data.keys()):
        if not seq_key.startswith("ACTION_SEQUENCE"):
            continue
        
        print(f"\nProcessing {seq_key}...")
        sequence = sparse_data[seq_key]
        print(f"  Sparse waypoints: {len(sequence)}")
        
        # Reset to home
        home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_cube")
        mujoco.mj_resetDataKeyframe(model, data_mj, home_key_id)
        mujoco.mj_forward(model, data_mj)
        
        dense_sequence = []
        
        # Record initial state
        dense_sequence.append(data_mj.qpos[:6].copy().tolist())
        
        # Execute each segment and record EVERY step
        for i in range(len(sequence) - 1):
            start_pose = np.array(sequence[i], dtype=np.float64)
            end_pose = np.array(sequence[i+1], dtype=np.float64)
            
            # Determine duration
            joint_dist = np.linalg.norm(start_pose[:5] - end_pose[:5])
            gripper_dist = abs(start_pose[-1] - end_pose[-1])
            
            if joint_dist < 0.01 and gripper_dist > 0.1:
                duration = GRIPPER_TIME
            else:
                duration = max(joint_dist / JOINT_SPEED, MIN_MOVE_TIME)
            
            # Simulate segment
            dt = model.opt.timestep
            steps = int(math.ceil(duration / dt))
            
            for s in range(steps):
                t = (s + 1) / steps
                smooth_t = t * t * (3 - 2 * t)
                target_cmd = start_pose + (end_pose - start_pose) * smooth_t
                
                # Apply control
                data_mj.ctrl[:6] = target_cmd
                
                # Step physics
                mujoco.mj_step(model, data_mj)
                
                # Record state every N steps
                if s % 3 == 0:  # Record every 3rd step to avoid TOO much data
                    dense_sequence.append(data_mj.qpos[:6].copy().tolist())
        
        print(f"  Dense waypoints: {len(dense_sequence)}")
        dense_data[seq_key] = dense_sequence
    
    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(dense_data, f, indent=2)
    
    print(f"\n✅ Done! Created {len(dense_data)} dense sequences.")
    print(f"   Average length: {np.mean([len(v) for v in dense_data.values()]):.1f} waypoints")
    print("=" * 60)

if __name__ == "__main__":
    densify_demos()

