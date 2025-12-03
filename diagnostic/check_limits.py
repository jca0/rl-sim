import json
import numpy as np
import mujoco
from pathlib import Path

# Paths
SCENE_PATH = Path("submodule/rl-sim/mujoco_sim/trs_so_arm100/scene.xml")
TELEOP_PATH = Path("submodule/rl-sim/mujoco_sim/teleop.json")

def main():
    print("="*60)
    print("DIAGNOSTICS: Checking Demo vs Robot Limits")
    print("="*60)

    # 1. Load Robot Limits
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    ctrl_low = model.actuator_ctrlrange[:, 0]
    ctrl_high = model.actuator_ctrlrange[:, 1]
    joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

    print(f"Robot Joint Limits (from XML):")
    for i, name in enumerate(joint_names):
        print(f"  {name:12s}: [{ctrl_low[i]:.4f}, {ctrl_high[i]:.4f}]")

    # 2. Load Demos
    with open(TELEOP_PATH) as f:
        data = json.load(f)
    
    sequences = []
    for key in data:
        if key.startswith("ACTION_SEQUENCE"):
            sequences.append(data[key])
            
    all_waypoints = np.concatenate(sequences, axis=0)
    print(f"\nLoaded {len(sequences)} sequences, {len(all_waypoints)} total waypoints.")

    # 3. Check for violations
    print("\nChecking for Out-of-Bounds commands...")
    violations = 0
    for i in range(6):
        vals = all_waypoints[:, i]
        min_val = vals.min()
        max_val = vals.max()
        
        status = "OK"
        if min_val < ctrl_low[i] - 0.001 or max_val > ctrl_high[i] + 0.001:
            status = "VIOLATION ❌"
            violations += 1
            
        print(f"  Joint {i} ({joint_names[i]}): Demo Range [{min_val:.4f}, {max_val:.4f}] -> {status}")
        if "VIOLATION" in status:
            print(f"    -> Low Limit Diff: {min_val - ctrl_low[i]:.4f}")
            print(f"    -> High Limit Diff: {max_val - ctrl_high[i]:.4f}")

    if violations > 0:
        print("\n⚠️  WARNING: The demos command positions OUTSIDE the robot's limits.")
        print("    This effectively changes the action during normalization (clipping).")
        print("    The policy learns to output 'Max', but the demo meant 'Max + epsilon'.")
    else:
        print("\n✅ Demos are strictly within robot limits.")

if __name__ == "__main__":
    main()

