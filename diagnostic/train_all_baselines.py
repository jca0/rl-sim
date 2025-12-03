#!/usr/bin/env python3
"""Train multiple baseline policies and compare results."""

import subprocess
import sys
from pathlib import Path

def run_training(script_name, exp_name, extra_args=""):
    """Run a training script."""
    cmd = f"python -m rl_env.{script_name} --exp-name {exp_name} --epochs 200 --device cuda --n-eval-episodes 5 {extra_args}"
    print(f"\n{'='*70}")
    print(f"TRAINING: {exp_name}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ {exp_name} failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✅ {exp_name} completed successfully")
        return True

def main():
    print("🚀 Training All Baseline Policies")
    print("=" * 70)
    print("This will train:")
    print("  1. Simple BC (MLP)")
    print("  2. ACT (Action Chunking Transformer)")
    print("=" * 70)
    
    results = {}
    
    # Train simple BC
    results['BC'] = run_training(
        "train_cube_robot_env_bc",
        "bc_baseline",
        "--teleop-path mujoco_sim/teleop.json"
    )
    
    # Train ACT
    results['ACT'] = run_training(
        "train_cube_robot_env_act",
        "act_baseline",
        "--teleop-path mujoco_sim/teleop.json --chunk-size 10"
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {name:20s} {status}")
    
    print("\n📹 Check videos in runs/*/videos/ to compare performance!")
    print("=" * 70)

if __name__ == "__main__":
    main()

