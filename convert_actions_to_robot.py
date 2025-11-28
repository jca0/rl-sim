#!/usr/bin/env python3
"""
Convert MuJoCo action sequences into SO-100 servo counts (0-4095).

The script reads:
  1. A JSON file that contains the simulated ACTION_SEQUENCE (list of lists).
  2. A calibration JSON with offsets, scales, directions, and sim rest poses.

It outputs joint targets as 12-bit servo counts, optionally writing them to a
new JSON file or stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

SIM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
ROBOT_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
SIM_TO_ROBOT = dict(zip(SIM_JOINT_NAMES, ROBOT_JOINT_NAMES))
MAX_SERVO_VALUE = 4095  # 12-bit Feetech servo range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate MuJoCo joint trajectories into robot servo counts.",
    )
    parser.add_argument(
        "--actions",
        required=True,
        type=Path,
        help="Path to JSON file that stores the simulated action sequence.",
    )
    parser.add_argument(
        "--calibration",
        default=Path(__file__).with_name("calibration.json"),
        type=Path,
        help="Path to calibration JSON (defaults to calibration.json next to this script).",
    )
    parser.add_argument(
        "--output",
        default="-",
        type=str,
        help="Optional path for the converted JSON. Use '-' for stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("list", "dict"),
        default="list",
        help="Output format: list of lists or list of dicts keyed by joint name.",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=MAX_SERVO_VALUE,
        help="Upper bound for servo counts (default: 4095).",
    )
    return parser.parse_args()


def load_action_sequence(path: Path) -> List[Sequence[float]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        for key in ("action_sequence", "actions", "sequence", "trajectory"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a list of joint targets or a dict containing one in {path}"
        )
    for idx, entry in enumerate(payload):
        if not isinstance(entry, (list, tuple)):
            raise ValueError(f"ACTION_SEQUENCE[{idx}] is not a list/tuple: {entry}")
        if len(entry) != len(SIM_JOINT_NAMES):
            raise ValueError(
                f"ACTION_SEQUENCE[{idx}] expected {len(SIM_JOINT_NAMES)} joints, "
                f"got {len(entry)}"
            )
    return payload


def load_calibration(path: Path) -> Dict:
    calib = json.loads(path.read_text())
    required = ("joint_names", "offsets", "scales", "directions")
    missing = [key for key in required if key not in calib]
    if missing:
        raise KeyError(f"Calibration file {path} missing keys: {', '.join(missing)}")
    n = len(calib["joint_names"])
    for key in required[1:]:
        if len(calib[key]) != n:
            raise ValueError(f"Calibration field '{key}' must have {n} entries.")
    return calib


def convert_action(
    sim_action: Sequence[float],
    calib: Dict,
    max_value: int,
    sim_rest: Dict[str, float],
) -> List[int]:
    joint_indices = {name: idx for idx, name in enumerate(calib["joint_names"])}
    offsets = calib["offsets"]
    scales = calib["scales"]
    directions = calib["directions"]

    servo_values: List[int] = []
    for sim_name, sim_value in zip(SIM_JOINT_NAMES, sim_action):
        robot_name = SIM_TO_ROBOT.get(sim_name)
        if robot_name is None:
            raise KeyError(f"No robot joint mapping for simulator joint '{sim_name}'.")
        if robot_name not in joint_indices:
            raise KeyError(
                f"Joint '{robot_name}' not found in calibration joint_names list."
            )

        idx = joint_indices[robot_name]
        rest = sim_rest.get(robot_name, 0.0)
        raw_value = offsets[idx] + directions[idx] * scales[idx] * (sim_value - rest)
        servo_value = max(0, min(max_value, int(round(raw_value))))
        servo_values.append(servo_value)

    return servo_values


def convert_sequence(
    actions: Iterable[Sequence[float]], calib: Dict, max_value: int
) -> List[List[int]]:
    sim_rest = calib.get("sim_rest_qpos", {})
    return [convert_action(action, calib, max_value, sim_rest) for action in actions]


def format_output(
    servo_sequence: List[List[int]], output_format: str
) -> Dict[str, Sequence]:
    if output_format == "dict":
        dict_sequence = [
            dict(zip(ROBOT_JOINT_NAMES, servo_values))
            for servo_values in servo_sequence
        ]
        return {"joint_names": ROBOT_JOINT_NAMES, "action_sequence": dict_sequence}
    return {"joint_names": ROBOT_JOINT_NAMES, "action_sequence": servo_sequence}


def dumps_with_inline_actions(payload: Dict[str, Sequence], indent: int = 2) -> str:
    """
    JSON serializer that keeps each action on one line for readability.
    """
    if "joint_names" not in payload or "action_sequence" not in payload:
        return json.dumps(payload, indent=indent)

    joint_names = payload["joint_names"]
    actions = payload["action_sequence"]
    indent1 = " " * indent
    indent2 = indent1 * 2
    lines = ["{"]

    lines.append(f'{indent1}"joint_names": [')
    for idx, name in enumerate(joint_names):
        comma = "," if idx < len(joint_names) - 1 else ""
        lines.append(f"{indent2}{json.dumps(name)}{comma}")
    lines.append(f"{indent1}],")

    lines.append(f'{indent1}"action_sequence": [')
    for idx, action in enumerate(actions):
        comma = "," if idx < len(actions) - 1 else ""
        action_str = json.dumps(action, separators=(", ", ": "))
        lines.append(f"{indent2}{action_str}{comma}")
    lines.append(f"{indent1}]")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    actions = load_action_sequence(args.actions)
    calibration = load_calibration(args.calibration)
    servo_sequence = convert_sequence(actions, calibration, args.max_value)
    formatted = format_output(servo_sequence, args.format)
    output_text = dumps_with_inline_actions(formatted, indent=2)

    if args.output == "-" or args.output == "":
        print(output_text)
    else:
        output_path = Path(args.output)
        output_path.write_text(output_text + "\n")
        print(f"Wrote {len(servo_sequence)} actions to {output_path}")


if __name__ == "__main__":
    main()

