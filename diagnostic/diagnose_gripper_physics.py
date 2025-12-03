#!/usr/bin/env python3
"""
Diagnose Gripper Physics: Does the gripper actually grip the cube?

Test sequence:
1. Reset to home position
2. Move end-effector directly above cube
3. Lower to cube height
4. Close gripper
5. Lift straight up
6. Check if cube lifts or falls

If cube falls → Physics is broken.
If cube lifts → Physics works, RL just needs better training.
"""

import json
import numpy as np
import gymnasium as gym
import mujoco
import imageio.v3 as iio
from pathlib import Path

import rl_env  # Registers RedCubePick-v0

def diagnose_gripper():
    print("=" * 60)
    print("GRIPPER PHYSICS DIAGNOSTIC")
    print("=" * 60)
    
    env = gym.make("RedCubePick-v0", render_mode="rgb_array")
    obs, info = env.reset()
    
    # Unwrap to get the actual env (gym.make wraps it in TimeLimit)
    base_env = env.unwrapped
    
    cube_pos = base_env.data.xpos[base_env.cube_body_id].copy()
    ee_pos = base_env.data.xpos[base_env.ee_body_id].copy()
    
    print(f"\n[INITIAL STATE]")
    print(f"  Cube position: {cube_pos}")
    print(f"  EE position:   {ee_pos}")
    print(f"  Distance:      {np.linalg.norm(ee_pos - cube_pos):.4f}m")
    
    frames = []
    
    # Phase 1: Move above cube (X, Y aligned, Z higher)
    print(f"\n[PHASE 1: Moving above cube]")
    target_above = cube_pos.copy()
    target_above[2] += 0.08  # 8cm above cube
    
    for step in range(50):
        ee_pos = base_env.data.xpos[base_env.ee_body_id].copy()
        error = target_above - ee_pos
        
        # Simple proportional control for arm joints
        # We'll just use a heuristic: move gripper open (-1.0)
        action = np.zeros(6)
        
        # Get current joint positions
        current_joints = base_env.data.qpos[:5].copy()
        
        # Compute desired joint positions using inverse kinematics (simplified)
        # For now, let's just use small increments toward target
        # This is a hack - proper IK would be better
        
        # Simple heuristic: adjust joints proportionally to error
        action[:5] = np.clip(error[:3].sum() * np.array([0.5, 0.3, 0.3, 0.2, 0.1]), -0.3, 0.3)
        action[5] = -1.0  # Gripper open
        
        obs, reward, term, trunc, info = env.step(action)
        frames.append(env.render())
        
        if step % 10 == 0:
            ee_pos = base_env.data.xpos[base_env.ee_body_id].copy()
            print(f"  Step {step:3d}: EE at {ee_pos}, dist={np.linalg.norm(ee_pos - target_above):.4f}m")
    
    # Phase 2: Lower to cube
    print(f"\n[PHASE 2: Lowering to cube]")
    target_at_cube = cube_pos.copy()
    target_at_cube[2] += 0.015  # Just above cube surface
    
    for step in range(30):
        action = np.zeros(6)
        action[:5] = np.array([0.0, 0.0, -0.2, 0.0, 0.0])  # Lower down
        action[5] = -1.0  # Gripper still open
        
        obs, reward, term, trunc, info = env.step(action)
        frames.append(env.render())
        
        if step % 10 == 0:
            ee_pos = base_env.data.xpos[base_env.ee_body_id].copy()
            cube_pos_now = base_env.data.xpos[base_env.cube_body_id].copy()
            print(f"  Step {step:3d}: EE at {ee_pos}, Cube at {cube_pos_now}")
    
    ee_pos = base_env.data.xpos[base_env.ee_body_id].copy()
    cube_pos_before_grasp = base_env.data.xpos[base_env.cube_body_id].copy()
    dist_to_cube = np.linalg.norm(ee_pos - cube_pos_before_grasp)
    
    print(f"\n[PRE-GRASP STATE]")
    print(f"  EE position:   {ee_pos}")
    print(f"  Cube position: {cube_pos_before_grasp}")
    print(f"  Distance:      {dist_to_cube:.4f}m")
    
    # Phase 3: CLOSE GRIPPER
    print(f"\n[PHASE 3: Closing gripper]")
    for step in range(20):
        action = np.zeros(6)
        action[:5] = 0.0  # Hold arm still
        action[5] = 1.0   # CLOSE GRIPPER
        
        obs, reward, term, trunc, info = env.step(action)
        frames.append(env.render())
        
        if step % 5 == 0:
            gripper_pos = base_env.data.qpos[5]
            print(f"  Step {step:3d}: Gripper position = {gripper_pos:.4f}")
    
    gripper_closed_pos = base_env.data.qpos[5]
    print(f"  Final gripper position: {gripper_closed_pos:.4f}")
    
    # Phase 4: LIFT UP
    print(f"\n[PHASE 4: Lifting]")
    cube_pos_before_lift = base_env.data.xpos[base_env.cube_body_id].copy()
    
    for step in range(50):
        action = np.zeros(6)
        action[:5] = np.array([0.0, 0.0, 0.3, 0.0, 0.0])  # Lift up
        action[5] = 1.0  # Keep gripper closed
        
        obs, reward, term, trunc, info = env.step(action)
        frames.append(env.render())
        
        if step % 10 == 0:
            cube_pos_now = base_env.data.xpos[base_env.cube_body_id].copy()
            ee_pos_now = base_env.data.xpos[base_env.ee_body_id].copy()
            print(f"  Step {step:3d}: Cube Z={cube_pos_now[2]:.4f}m, EE Z={ee_pos_now[2]:.4f}m")
    
    # Final check
    cube_pos_after_lift = base_env.data.xpos[base_env.cube_body_id].copy()
    lift_height = cube_pos_after_lift[2] - cube_pos_before_lift[2]
    
    print(f"\n[FINAL RESULT]")
    print(f"  Cube before lift: {cube_pos_before_lift}")
    print(f"  Cube after lift:  {cube_pos_after_lift}")
    print(f"  Lift height:      {lift_height:.4f}m")
    
    # Save video
    video_path = Path("runs/gripper_diagnostic.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(video_path), frames, fps=30)
    print(f"\n✅ Video saved to: {video_path}")
    
    # Verdict
    print(f"\n{'='*60}")
    if lift_height > 0.03:
        print("✅ VERDICT: Gripper physics WORKS! Cube lifted successfully.")
        print("   → Problem is RL training, not physics.")
    elif lift_height > 0.01:
        print("⚠️  VERDICT: Gripper is WEAK. Cube moved slightly but didn't lift.")
        print("   → Need to boost gripper force or improve contact.")
    else:
        print("❌ VERDICT: Gripper physics BROKEN. Cube didn't move.")
        print("   → Fix physics parameters before training RL.")
    print(f"{'='*60}")
    
    env.close()

if __name__ == "__main__":
    diagnose_gripper()

