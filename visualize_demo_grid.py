#!/usr/bin/env python3
"""
Create video grids showing multiple demos at once for easy comparison.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

SCENE_PATH = Path("mujoco_sim/trs_so_arm100/scene.xml")
OUTPUT_DIR = Path("runs/demo_grid_videos")

import sys
if len(sys.argv) > 1:
    DEMO_PATH = Path(sys.argv[1])
else:
    DEMO_PATH = Path("mujoco_sim/teleop_dense.json")

JOINT_SPEED = 1.5
MIN_MOVE_TIME = 0.2
GRIPPER_TIME = 0.5

def render_sequence(model, data_mj, renderer, sequence, seq_name):
    """Simulate and render a sequence (SAME LOGIC AS visualize_demo.py)."""
    
    # Reset to home
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_cube")
    mujoco.mj_resetDataKeyframe(model, data_mj, home_key_id)
    mujoco.mj_forward(model, data_mj)
    
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    cube_initial_z = data_mj.xpos[cube_body_id][2]
    
    frames = []
    max_cube_z = cube_initial_z
    
    # Render initial state
    renderer.update_scene(data_mj)
    frames.append(renderer.render().copy())
    
    # Execute each segment (SAME AS visualize_demo.py)
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
            # Smooth interpolation
            smooth_t = t * t * (3 - 2 * t)
            target_cmd = start_pose + (end_pose - start_pose) * smooth_t
            
            # Apply control
            data_mj.ctrl[:6] = target_cmd
            
            # Step physics
            mujoco.mj_step(model, data_mj)
            
            # Track max cube height
            cube_z = data_mj.xpos[cube_body_id][2]
            max_cube_z = max(max_cube_z, cube_z)
            
            # Render every few steps (for smaller grid videos)
            if s % 5 == 0 or s == steps - 1:
                renderer.update_scene(data_mj)
                frame = renderer.render()
                frames.append(frame)
    
    lift_height = max_cube_z - cube_initial_z
    
    return frames, lift_height

def add_label_to_frame(frame, text, color=(255, 255, 0)):
    """Add text label to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Use default font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw text with background
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Background rectangle
    draw.rectangle([(5, 5), (5 + text_width + 10, 5 + text_height + 10)], fill=(0, 0, 0, 180))
    
    # Text
    draw.text((10, 10), text, fill=color, font=font)
    
    return np.array(img)

def create_grid(frames_list, labels, grid_size=(4, 4)):
    """Create a grid of videos."""
    rows, cols = grid_size
    
    # Ensure all frame sequences have the same length (pad if needed)
    max_len = max(len(f) for f in frames_list)
    
    padded_frames = []
    for frames in frames_list:
        if len(frames) < max_len:
            # Repeat last frame
            frames = frames + [frames[-1]] * (max_len - len(frames))
        padded_frames.append(frames)
    
    # Create grid for each timestep
    h, w = padded_frames[0][0].shape[:2]
    grid_frames = []
    
    for t in range(max_len):
        # Create empty grid
        grid_h = h * rows
        grid_w = w * cols
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Fill grid
        for idx in range(min(len(padded_frames), rows * cols)):
            row = idx // cols
            col = idx % cols
            
            frame = padded_frames[idx][t]
            
            # Add label
            frame_labeled = add_label_to_frame(frame, labels[idx])
            
            # Place in grid
            y_start = row * h
            x_start = col * w
            grid[y_start:y_start+h, x_start:x_start+w] = frame_labeled
        
        grid_frames.append(grid)
    
    return grid_frames

def main():
    print("=" * 60)
    print("CREATING DEMO GRID VIDEOS")
    print(f"Demo file: {DEMO_PATH}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load demos
    with open(DEMO_PATH) as f:
        data = json.load(f)
    
    sequences = []
    for key in sorted(data.keys()):
        if key.startswith("ACTION_SEQUENCE"):
            sequences.append((key, np.array(data[key], dtype=np.float64)))
    
    print(f"Found {len(sequences)} sequences")
    
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)  # Smaller for grid
    
    # Boost gripper
    for i in range(model.nu):
        if "Jaw" in model.actuator(i).name:
            model.actuator_gainprm[i, 0] = 500.0
            model.actuator_biasprm[i, 1] = -500.0
    
    # Process in batches of 16 (4x4 grid)
    batch_size = 16
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sequences))
        batch = sequences[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (sequences {start_idx} to {end_idx-1})...")
        
        frames_list = []
        labels = []
        
        for seq_name, sequence in batch:
            print(f"  Rendering {seq_name}...")
            frames, lift_height = render_sequence(model, data_mj, renderer, sequence, seq_name)
            
            # Create label with lift info
            seq_num = seq_name.split("_")[-1]
            label = f"#{seq_num} lift={lift_height:.3f}m"
            
            frames_list.append(frames)
            labels.append(label)
        
        # Create grid
        print(f"  Creating grid...")
        grid_frames = create_grid(frames_list, labels, grid_size=(4, 4))
        
        # Save video
        demo_name = DEMO_PATH.stem  # e.g., "teleop_dense" or "teleop"
        output_path = OUTPUT_DIR / f"{demo_name}_grid_batch{batch_idx + 1}.mp4"
        print(f"  Saving to {output_path}...")
        iio.imwrite(str(output_path), grid_frames, fps=30, codec="libx264", quality=7)
        
        print(f"  ✅ Saved {output_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Done! Created {num_batches} grid videos in {OUTPUT_DIR}/")
    print(f"   Each video shows 16 demos in a 4x4 grid.")
    print(f"   Watch them and tell me which sequences actually lift the cube!")
    print("=" * 60)
    
    if hasattr(renderer, 'close'):
        renderer.close()

if __name__ == "__main__":
    main()

