#!/usr/bin/env python3
"""
Diagnose Gripper Physics V2: Use actual demo waypoints to test grip.

This version:
1. Loads a successful demo from teleop_dense.json
2. Plays back the demo waypoints DIRECTLY (like visualize_demo.py)
3. Checks if the cube actually lifts

If cube lifts → Physics works, demos are valid
If cube doesn't lift → Physics is broken OR demos are fake
"""

import json
import math
import numpy as np
import mujoco
import imageio.v3 as iio
from pathlib import Path

SCENE_PATH = Path("mujoco_sim/trs_so_arm100/scene.xml")
JOINT_SPEED = 1.5  # rad/s
MIN_MOVE_TIME = 0.2
GRIPPER_TIME = 0.5

def diagnose_gripper_with_demo():
    print("=" * 60)
    print("GRIPPER PHYSICS DIAGNOSTIC V2 (Using Real Demo)")
    print("=" * 60)
    
    # Load demo
    demo_path = Path("mujoco_sim/teleop_dense.json")
    with open(demo_path, 'r') as f:
        demos = json.load(f)
    
    demo_key = "ACTION_SEQUENCE_0"
    sequence = demos[demo_key]
    
    print(f"\n[DEMO INFO]")
    print(f"  Demo: {demo_key}")
    print(f"  Waypoints: {len(sequence)}")
    for i, waypoint in enumerate(sequence):
        print(f"  Waypoint {i}: {waypoint}")
    
    # Setup MuJoCo (EXACTLY like visualize_demo.py)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # BOOST GRIPPER (same as env does)
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        if "Jaw" in actuator_name:
            model.actuator_gainprm[i, 0] = 500.0
            model.actuator_biasprm[i, 1] = -500.0
            print(f"\n[GRIPPER BOOST] {actuator_name}: Kp=500")
    
    # Reset to home
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_cube")
    mujoco.mj_resetDataKeyframe(model, data_mj, home_key_id)
    mujoco.mj_forward(model, data_mj)
    
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    cube_pos_initial = data_mj.xpos[cube_body_id].copy()
    
    print(f"\n[INITIAL STATE]")
    print(f"  Cube position: {cube_pos_initial}")
    
    frames = []
    cube_height_history = [cube_pos_initial[2]]
    
    # Render initial
    renderer.update_scene(data_mj)
    frames.append(renderer.render().copy())
    
    print(f"\n[PLAYBACK - Direct Control Like visualize_demo.py]")
    
    # Execute each segment (EXACTLY like visualize_demo.py)
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
            smooth_t = t * t * (3 - 2 * t)
            target_cmd = start_pose + (end_pose - start_pose) * smooth_t
            
            # DIRECT CONTROL (like visualize_demo.py)
            data_mj.ctrl[:6] = target_cmd
            
            # Step physics
            mujoco.mj_step(model, data_mj)
            
            # Track cube height
            cube_pos = data_mj.xpos[cube_body_id].copy()
            cube_height_history.append(cube_pos[2])
            
            # Render
            if s % 5 == 0 or s == steps - 1:
                renderer.update_scene(data_mj)
                frames.append(renderer.render().copy())
        
        # Report after each waypoint
        cube_pos = data_mj.xpos[cube_body_id].copy()
        ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw")
        ee_pos = data_mj.xpos[ee_body_id].copy()
        gripper_pos = data_mj.qpos[5]
        
        print(f"    → Cube Z={cube_pos[2]:.4f}m, EE Z={ee_pos[2]:.4f}m, Gripper={gripper_pos:.3f}")
    
    # Final check
    cube_pos_final = data_mj.xpos[cube_body_id].copy()
    max_height = max(cube_height_history)
    lift_height = max_height - cube_pos_initial[2]
    
    print(f"\n[FINAL RESULT]")
    print(f"  Cube initial Z: {cube_pos_initial[2]:.4f}m")
    print(f"  Cube final Z:   {cube_pos_final[2]:.4f}m")
    print(f"  Max height:     {max_height:.4f}m")
    print(f"  Lift height:    {lift_height:.4f}m")
    
    # Save video
    video_path = Path("runs/gripper_diagnostic_v2.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(video_path), frames, fps=30)
    print(f"\n✅ Video saved to: {video_path}")
    
    # Verdict
    print(f"\n{'='*60}")
    if lift_height > 0.03:
        print("✅ VERDICT: Gripper physics WORKS! Cube lifted successfully.")
        print("   → Demos are valid. Problem is RL training, not physics.")
        print("   → CONCLUSION: Need better RL algorithm or training strategy.")
    elif lift_height > 0.01:
        print("⚠️  VERDICT: Gripper is WEAK. Cube moved slightly but didn't lift.")
        print("   → Need to boost gripper force or improve contact.")
    else:
        print("❌ VERDICT: Gripper physics BROKEN or demos are invalid.")
        print("   → Either fix physics OR demos don't actually work.")
    print(f"{'='*60}")
    
    if hasattr(renderer, 'close'):
        renderer.close()

if __name__ == "__main__":
    diagnose_gripper_with_demo()

