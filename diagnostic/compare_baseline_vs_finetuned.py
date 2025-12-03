"""
Compare baseline vs fine-tuned policy gripper behavior.
This will show if fine-tuning reduced gripper force.
"""

import sys
from pathlib import Path
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodule" / "rl-sim"))

from rl_env.train_trajectory_tracker import TrajectoryPolicy

print("\n" + "="*70)
print("COMPARING BASELINE VS FINE-TUNED GRIPPER BEHAVIOR")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load baseline policy
baseline_path = REPO_ROOT / "submodule" / "rl-sim" / "runs" / "FINAL_FIXED__1764557628" / "policy_epoch2000.pt"
baseline_policy = TrajectoryPolicy().to(device)
baseline_policy.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=False))
baseline_policy.eval()
print(f"\n✓ Baseline loaded: {baseline_path.name}")

# Load fine-tuned policy
finetuned_runs = sorted((REPO_ROOT / "submodule" / "rl-sim" / "runs").glob("FINETUNED_OPTIMIZED_*"), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
if not finetuned_runs:
    print("❌ No fine-tuned policy found!")
    sys.exit(1)

finetuned_path = finetuned_runs[0] / "policy_finetuned.pt"
finetuned_policy = TrajectoryPolicy().to(device)
finetuned_policy.load_state_dict(torch.load(finetuned_path, map_location=device, weights_only=False))
finetuned_policy.eval()
print(f"✓ Fine-tuned loaded: {finetuned_path.parent.name}/{finetuned_path.name}")

# Compare gripper trajectories
print("\n" + "="*70)
print("GRIPPER POSITION COMPARISON")
print("="*70)

baseline_gripper = []
finetuned_gripper = []

for step in range(200):
    time_norm = torch.FloatTensor([[step / 200.0]]).to(device)
    
    with torch.no_grad():
        baseline_joints = baseline_policy(time_norm).cpu().numpy()[0]
        finetuned_joints = finetuned_policy(time_norm).cpu().numpy()[0]
    
    baseline_gripper.append(baseline_joints[5])
    finetuned_gripper.append(finetuned_joints[5])

baseline_gripper = np.array(baseline_gripper)
finetuned_gripper = np.array(finetuned_gripper)

# Analysis
print(f"\nBaseline gripper:")
print(f"  Max position: {baseline_gripper.max():.4f} rad")
print(f"  Mean position: {baseline_gripper.mean():.4f} rad")
print(f"  When closing (>0.5): {(baseline_gripper > 0.5).sum()} steps")

print(f"\nFine-tuned gripper:")
print(f"  Max position: {finetuned_gripper.max():.4f} rad")
print(f"  Mean position: {finetuned_gripper.mean():.4f} rad")
print(f"  When closing (>0.5): {(finetuned_gripper > 0.5).sum()} steps")

print(f"\nDifference:")
print(f"  Max position: {finetuned_gripper.max() - baseline_gripper.max():.4f} rad")
print(f"  Mean position: {finetuned_gripper.mean() - baseline_gripper.mean():.4f} rad")

reduction_pct = (baseline_gripper.max() - finetuned_gripper.max()) / baseline_gripper.max() * 100

print("\n" + "="*70)
if finetuned_gripper.max() < baseline_gripper.max():
    print(f"✅ FINE-TUNED GRIPS LESS! ({reduction_pct:.1f}% reduction)")
    print("   This should prevent overload on real robot!")
elif finetuned_gripper.max() > baseline_gripper.max():
    print(f"❌ FINE-TUNED GRIPS MORE! ({-reduction_pct:.1f}% increase)")
    print("   This will make overload worse!")
else:
    print("⚠️  NO CHANGE in gripper behavior")
    print("   Fine-tuning didn't adapt the policy")

print("="*70)

