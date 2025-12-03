import json
import numpy as np
import gymnasium as gym
from pathlib import Path
import rl_env
from rl_env.train_ppo_transformer import joint_pos_to_normalized_action

def verify_action_logic():
    print("="*60)
    print("SCIENTIFIC VERIFICATION: Action -> Env -> State")
    print("="*60)

    # 1. Load Env
    env = gym.make("RedCubePick-v0")
    base_env = env.unwrapped
    
    print(f"Env Control Limits:")
    print(f"  Low:  {base_env.ctrl_low}")
    print(f"  High: {base_env.ctrl_high}")

    # 2. Load a Demo Target
    teleop_path = Path("mujoco_sim/teleop.json")
    with open(teleop_path) as f:
        data = json.load(f)
    
    # Get the second waypoint of the first demo (usually a movement)
    # Waypoint 0 is usually home, Waypoint 1 is the first reach
    target_joints = np.array(data["ACTION_SEQUENCE_0"][1])
    print(f"\nTarget Joint Configuration (from Demo):")
    print(f"  {target_joints}")

    # 3. Calculate Normalized Action
    print("\nCalculating Normalized Action...")
    action = joint_pos_to_normalized_action(
        target_joints, 
        base_env.ctrl_low, 
        base_env.ctrl_high
    )
    print(f"  Action: {action}")

    # 4. Apply Action to Env
    print("\nApplying Action to Environment...")
    env.reset()
    
    # Step enough times for the PD controller to settle (e.g., 1 second)
    # Frame skip is 3, timestep 0.002 -> dt = 0.006
    # 1 second / 0.006 ~= 166 steps
    for _ in range(100):
        obs, _, _, _, _ = env.step(action)
    
    # 5. Measure Result
    final_qpos = env.unwrapped.data.qpos[:6]
    print(f"\nFinal Robot Joint State:")
    print(f"  {final_qpos}")

    # 6. Compare
    diff = final_qpos - target_joints
    mse = np.mean(diff**2)
    print(f"\nDifference (Actual - Target):")
    print(f"  {diff}")
    print(f"  MSE: {mse:.6f}")

    if mse > 1e-3:
        print("\n❌ FAIL: Significant mismatch detected.")
        print("   The environment did NOT go where the demo commanded.")
        
        # Diagnose clipping
        clipped = False
        for i in range(6):
            if target_joints[i] < base_env.ctrl_low[i] or target_joints[i] > base_env.ctrl_high[i]:
                print(f"   -> Joint {i} target {target_joints[i]:.3f} is OUTSIDE limits [{base_env.ctrl_low[i]:.3f}, {base_env.ctrl_high[i]:.3f}]")
                clipped = True
        
        if not clipped:
            print("   -> Limits were NOT violated. Error is likely in PD gains or normalization math.")
    else:
        print("\n✅ SUCCESS: Action logic is sound. Robot went exactly to target.")

if __name__ == "__main__":
    verify_action_logic()

