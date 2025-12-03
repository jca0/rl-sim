"""Check baseline policy gripper values."""

import sys
from pathlib import Path
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodule" / "rl-sim"))

from rl_env.train_trajectory_tracker import TrajectoryPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load baseline
baseline_path = REPO_ROOT / "submodule" / "rl-sim" / "runs" / "FINAL_FIXED__1764557628" / "policy_epoch2000.pt"
policy = TrajectoryPolicy().to(device)
policy.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=False))
policy.eval()

print("\n" + "="*70)
print("BASELINE POLICY GRIPPER ANALYSIS")
print("="*70)

gripper_values = []

for step in range(200):
    time_norm = torch.FloatTensor([[step / 200.0]]).to(device)
    
    with torch.no_grad():
        joints = policy(time_norm).cpu().numpy()[0]
    
    gripper_values.append(joints[5])
    
    if step % 40 == 0:
        print(f"Step {step:3d}: gripper = {joints[5]:.4f} rad")

gripper_values = np.array(gripper_values)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Max gripper position: {gripper_values.max():.4f} rad")
print(f"Mean gripper position: {gripper_values.mean():.4f} rad")
print(f"Min gripper position: {gripper_values.min():.4f} rad")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

if gripper_values.max() > 1.5:
    print(f"⚠️  Gripper closes to {gripper_values.max():.4f} rad (VERY TIGHT!)")
    print("   This is likely causing overload on real robot.")
    print(f"   Gripper range: [-0.174, 1.75] rad")
    print(f"   Baseline uses: {gripper_values.max() / 1.75 * 100:.1f}% of max closure")
else:
    print(f"✓ Gripper closes to {gripper_values.max():.4f} rad (moderate)")

print("="*70)

