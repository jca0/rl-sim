"""
Visualize a saved ACTION_SEQUENCE in MuJoCo.

Usage:
    python mujoco_sim/playback_action_sequence.py \
        --sequence-path runs/<run_name>/action_sequence.py

The sequence file must define ACTION_SEQUENCE exactly like
`mujoco_sim/mujoco_pick_place.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import mujoco
import mujoco.viewer as viewer
import numpy as np

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def load_action_sequence(sequence_path: Path) -> Sequence[Tuple[float, ...]]:
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")

    spec = importlib.util.spec_from_file_location("rl_action_sequence", sequence_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ACTION_SEQUENCE from {sequence_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["rl_action_sequence"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "ACTION_SEQUENCE"):
        raise AttributeError(
            f"{sequence_path} is missing ACTION_SEQUENCE. "
            "Ensure it matches mujoco_pick_place.py format."
        )

    actions = getattr(module, "ACTION_SEQUENCE")
    if not isinstance(actions, Iterable):
        raise TypeError("ACTION_SEQUENCE must be an iterable of joint tuples.")

    parsed: list[Tuple[float, ...]] = []
    for idx, entry in enumerate(actions):
        if not isinstance(entry, (tuple, list)) or len(entry) != len(JOINT_NAMES):
            raise ValueError(
                f"ACTION_SEQUENCE[{idx}] must be length {len(JOINT_NAMES)} tuple."
            )
        parsed.append(tuple(float(val) for val in entry))
    if len(parsed) == 0:
        raise ValueError("ACTION_SEQUENCE is empty.")
    return parsed


def run_sequence(
    model_path: Path,
    sequence: Sequence[Tuple[float, ...]],
    duration: float,
    loop: bool = False,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest_with_cube")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

    data.ctrl[:] = data.qpos[: len(JOINT_NAMES)]

    with viewer.launch_passive(model, data) as gui:
        seq_idx = 0
        last_change = time.time()
        start_qpos = data.ctrl[: len(JOINT_NAMES)].copy()
        target_qpos = np.array(sequence[0], dtype=np.float64)

        print(f"Loaded sequence ({len(sequence)} steps). Press ESC to exit.")
        while gui.is_running():
            step_start = time.time()
            now = time.time()
            progress = min((now - last_change) / max(duration, 1e-6), 1.0)
            t = progress * progress * (3 - 2 * progress)  # smoothstep
            current_cmd = start_qpos + (target_qpos - start_qpos) * t
            data.ctrl[: len(JOINT_NAMES)] = current_cmd

            if progress >= 1.0:
                seq_idx += 1
                if seq_idx >= len(sequence):
                    if loop:
                        seq_idx = 0
                    else:
                        print("Sequence complete. Holding final position.")
                        target_qpos = start_qpos = data.ctrl[
                            : len(JOINT_NAMES)
                        ].copy()
                        duration = 10.0
                        continue
                start_qpos = data.ctrl[: len(JOINT_NAMES)].copy()
                target_qpos = np.array(sequence[seq_idx], dtype=np.float64)
                last_change = now
                print(f"Step {seq_idx+1}/{len(sequence)}: {target_qpos}")

            mujoco.mj_step(model, data)
            gui.sync()

            time_until = model.opt.timestep - (time.time() - step_start)
            if time_until > 0:
                time.sleep(time_until)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playback RL action sequence in MuJoCo.")
    parser.add_argument(
        "--sequence-path",
        type=Path,
        required=True,
        help="Path to action_sequence.py file exported by train_pick_place.py",
    )
    default_model = (
        Path(__file__).resolve().parent / "trs_so_arm100" / "scene.xml"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help=f"MuJoCo scene xml (default: {default_model})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Seconds spent interpolating between consecutive steps.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the sequence indefinitely instead of stopping at the end.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = load_action_sequence(args.sequence_path)
    run_sequence(args.model_path, sequence, args.duration, args.loop)


if __name__ == "__main__":
    main()

