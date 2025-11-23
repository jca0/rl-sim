import time
import mujoco
from mujoco import MjModel, MjData, mj_resetDataKeyframe
import mujoco.viewer as viewer
import numpy as np

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
JAW_OPEN = 1.75
JAW_CLOSED = 0.17

# list of tuples of joint positions to move to
ACTION_SEQUENCE = [
    (0, -1.57, 1.57, 1.57, 1.57, JAW_OPEN),
    (-0.15, -2.10, 2.30, 1.05, 1.20, JAW_OPEN),
    (-0.15, -1.9, 2.1, 0.85, 1.20, JAW_OPEN),
    (-0.15, -1.6, 2.1, 0.85, 1.20, JAW_OPEN),
    (-0.15, -1.6, 2.1, 0.85, 1.8, JAW_CLOSED),
    (-0.15, -2, 2.1, 0.85, 1.8, JAW_CLOSED),
    (0.3, -2, 2.1, 0.85, 1.8, JAW_CLOSED),
    (0.3, -2, 2.1, 0.85, 1.8, JAW_OPEN),
]

def main():
    # Load the model and data
    model = mujoco.MjModel.from_xml_path("trs_so_arm100/scene.xml")
    data = mujoco.MjData(model)

    # Reset to the "home_with_cube" keyframe to ensure consistent start
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest_with_cube")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data) # Update physics state
    
    # Initialize control signals to current positions so it doesn't jerk
    data.ctrl[:] = data.qpos[:6]

    print("\nStarting Pick and Place Script...")

    with mujoco.viewer.launch_passive(model, data) as gui:
        start_time = time.time()
        
        # Sequence management
        seq_idx = 0
        last_seq_change = start_time
        
        # Movement state
        start_qpos = data.qpos[:6].copy()
        target_qpos = start_qpos.copy()
        duration = 1.0
        
        # Load first action
        if len(ACTION_SEQUENCE) > 0:
            target_qpos = ACTION_SEQUENCE[0]
            print(f"Action: {target_qpos}")

        while gui.is_running():
            step_start = time.time()
            
            # --- 1. LOGIC & INTERPOLATION ---
            now = time.time()
            elapsed = now - last_seq_change
            progress = min(elapsed / duration, 1.0)
            
            # Interpolate (Smooth movement)
            # smooth_step function: 3x^2 - 2x^3
            t = progress * progress * (3 - 2 * progress)
            current_cmd = start_qpos + (target_qpos - start_qpos) * t
            
            # Apply to actuators
            data.ctrl[:6] = current_cmd

            # Check if action is done
            if progress >= 1.0:
                if seq_idx < len(ACTION_SEQUENCE) - 1:
                    seq_idx += 1
                    # Prepare next action
                    start_qpos = data.ctrl[:6].copy() # Start from where we commanded
                    target_qpos = ACTION_SEQUENCE[seq_idx]
                    last_seq_change = now
                    print(f"Action: {target_qpos}")
                else:
                    # End of sequence, just hold final position
                    pass

            # --- 2. PHYSICS STEP ---
            mujoco.mj_step(model, data)
            gui.sync()

            # --- 3. LOGGING HELPER ---
            # Print current positions every second to help you calibrate
            if (now - start_time) % 2.0 < 0.02:
                # Format for easy copy-pasting into Python
                arr_str = ", ".join([f"{x:.3f}" for x in data.qpos[:6]])
                print(f"Current Joint QPos: np.array([{arr_str}])", flush=True)

            # --- 4. TIMING ---
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()