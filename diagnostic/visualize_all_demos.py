#!/usr/bin/env python3
"""
Visualize ALL demo sequences and report which ones successfully lift the cube.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import imageio.v3 as iio

SCENE_PATH = Path("mujoco_sim/trs_so_arm100/scene.xml")
DEMO_PATH = Path("mujoco_sim/teleop_SUPER_dense.json")
OUTPUT_DIR = Path("runs/demo_analysis")

JOINT_SPEED = 1.5
MIN_MOVE_TIME = 0.2
GRIPPER_TIME = 0.5

def analyze_sequence(model, data_mj, sequence, seq_name):
    """Simulate a sequence and check if it lifts the cube."""
    
    # Reset to home
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_cube")
    mujoco.mj_resetDataKeyframe(model, data_mj, home_key_id)
    mujoco.mj_forward(model, data_mj)
    
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    cube_initial_z = data_mj.xpos[cube_body_id][2]
    
    frames = []
    cube_heights = []
    
    # Execute sequence (simplified - just set positions directly)
    for waypoint in sequence:
        # Apply control
        data_mj.ctrl[:6] = waypoint
        
        # Step physics multiple times
        for _ in range(20):
            mujoco.mj_step(model, data_mj)
        
        cube_z = data_mj.xpos[cube_body_id][2]
        cube_heights.append(cube_z)
    
    max_height = max(cube_heights)
    lift_height = max_height - cube_initial_z
    success = lift_height > 0.03  # Lifted > 3cm
    
    return {
        'name': seq_name,
        'initial_z': cube_initial_z,
        'max_z': max_height,
        'lift_height': lift_height,
        'success': success
    }

def main():
    print("=" * 60)
    print("ANALYZING ALL DEMO SEQUENCES")
    print("=" * 60)
    
    # Load demos
    with open(DEMO_PATH) as f:
        data = json.load(f)
    
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    
    # Boost gripper
    for i in range(model.nu):
        if "Jaw" in model.actuator(i).name:
            model.actuator_gainprm[i, 0] = 500.0
            model.actuator_biasprm[i, 1] = -500.0
    
    results = []
    
    for seq_key in sorted(data.keys()):
        if not seq_key.startswith("ACTION_SEQUENCE"):
            continue
        
        sequence = np.array(data[seq_key], dtype=np.float64)
        result = analyze_sequence(model, data_mj, sequence, seq_key)
        results.append(result)
        
        status = "✅ SUCCESS" if result['success'] else "❌ FAIL"
        print(f"{seq_key}: {status} (lift={result['lift_height']:.4f}m, max_z={result['max_z']:.4f}m)")
    
    # Summary
    successes = [r for r in results if r['success']]
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total sequences: {len(results)}")
    print(f"  Successful: {len(successes)} ({100*len(successes)/len(results):.1f}%)")
    print(f"  Failed: {len(results) - len(successes)}")
    
    if successes:
        print(f"\n✅ SUCCESSFUL SEQUENCES:")
        for r in successes:
            print(f"  - {r['name']}: lift={r['lift_height']:.4f}m")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

