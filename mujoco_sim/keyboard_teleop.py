#!/usr/bin/env python3
"""
Keyboard teleoperation for the SO-100 MuJoCo scene using simple Cartesian IK.

Focus the small pygame window to send commands. Arrow keys translate the
end-effector in the table plane, A/Z move it along +Z/-Z, and SPACE toggles
the gripper. The MuJoCo viewer continues to run for visualization.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from multiprocessing import Pipe, Process
from typing import Set

try:
    import mujoco
    import mujoco.viewer as viewer
except ImportError as exc:  # pragma: no cover - loads at runtime
    raise ImportError(
        "keyboard_teleop.py requires the 'mujoco' package. "
        "Install mujoco>=3.0 to run this script."
    ) from exc

import numpy as np
import pygame

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
ARM_JOINT_COUNT = len(JOINT_NAMES) - 1  # exclude gripper
JAW_OPEN = 1.75
JAW_CLOSED = 0.17
EEF_BODY_NAME = "Moving_Jaw"

MODEL_PATH = (
    Path(__file__).resolve().parent / "trs_so_arm100" / "scene.xml"
)

# Key -> Cartesian direction (x, y, z)
CARTESIAN_BINDINGS = {
    pygame.K_DOWN: np.array([1.0, 0.0, 0.0]),      # +X (forward)
    pygame.K_UP: np.array([-1.0, 0.0, 0.0]),       # -X (backward)
    pygame.K_RIGHT: np.array([0.0, 1.0, 0.0]),     # +Y (left)
    pygame.K_LEFT: np.array([0.0, -1.0, 0.0]),     # -Y (right)
    pygame.K_a: np.array([0.0, 0.0, 1.0]),         # +Z (up)
    pygame.K_z: np.array([0.0, 0.0, -1.0]),        # -Z (down)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate the SO-100 end-effector in Cartesian space."
    )
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=0.04,
        help="End-effector translation speed in meters/second (default: 0.04).",
    )
    parser.add_argument(
        "--fast-multiplier",
        type=float,
        default=3.0,
        help="Multiplier applied while holding SHIFT (default: 3x).",
    )
    parser.add_argument(
        "--ik-iters",
        type=int,
        default=60,
        help="Maximum IK iterations per control step (default: 60).",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=1e-3,
        help="Damping used for the IK least-squares solve (default: 1e-3).",
    )
    parser.add_argument(
        "--max-hz",
        type=int,
        default=120,
        help="Upper bound on teleop update rate (default: 120 Hz).",
    )
    return parser.parse_args()


def maybe_reset_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest_with_cube")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)


def build_joint_limits(model: mujoco.MjModel, count: int) -> np.ndarray:
    limits = np.zeros((count, 2), dtype=np.float64)
    for i in range(count):
        if i < model.njnt:
            limits[i] = model.jnt_range[i]
        else:
            limits[i] = (-np.pi, np.pi)
    return limits


def solve_position_ik(
    model: mujoco.MjModel,
    reference_data: mujoco.MjData,
    scratch: mujoco.MjData,
    joint_guess: np.ndarray,
    target_pos: np.ndarray,
    joint_limits: np.ndarray,
    body_id: int,
    max_iters: int,
    tol: float,
    damping: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Run damped least-squares IK to reach the desired end-effector position."""
    scratch.qpos[:] = reference_data.qpos
    scratch.qvel[:] = reference_data.qvel
    scratch.act[:] = reference_data.act
    scratch.ctrl[:] = reference_data.ctrl
    scratch.qpos[: ARM_JOINT_COUNT] = joint_guess
    achieved_pos = scratch.xpos[body_id].copy()

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    for _ in range(max_iters):
        mujoco.mj_forward(model, scratch)
        achieved_pos = scratch.xpos[body_id].copy()
        err = target_pos - achieved_pos
        if np.linalg.norm(err) < tol:
            return scratch.qpos[: ARM_JOINT_COUNT].copy(), achieved_pos

        mujoco.mj_jacBody(model, scratch, jacp, None, body_id)
        J = jacp[:, : ARM_JOINT_COUNT]
        JT = J.T
        JJ_T = J @ JT + damping * np.eye(3)
        try:
            dq = JT @ np.linalg.solve(JJ_T, err)
        except np.linalg.LinAlgError:
            break

        scratch.qpos[: ARM_JOINT_COUNT] += dq
        for j in range(ARM_JOINT_COUNT):
            low, high = joint_limits[j]
            scratch.qpos[j] = np.clip(scratch.qpos[j], low, high)

    return None, achieved_pos


def pygame_loop(conn) -> None:
    pygame.init()
    screen = pygame.display.set_mode((620, 320))
    pygame.display.set_caption("SO-100 keyboard teleop (focus here)")
    font = pygame.font.SysFont("Menlo", 17)
    clock = pygame.time.Clock()

    pressed: Set[int] = set()
    joint_state = np.zeros(ARM_JOINT_COUNT, dtype=np.float64)
    eef_state = np.zeros(3, dtype=np.float64)
    gripper_closed = False
    running = True

    def draw_overlay() -> None:
        screen.fill((20, 20, 20))
        lines = [
            "SO-100 Keyboard Teleop",
            "Arrow keys: translate X/Y | A/Z: translate Z | SHIFT: 3x speed",
            "SPACE: toggle gripper | ESC: quit",
            f"Gripper: {'Closed' if gripper_closed else 'Open'}",
        ]
        lines.append(
            f"EEF target (m): {eef_state[0]:6.3f}, {eef_state[1]:6.3f}, {eef_state[2]:6.3f}"
        )
        lines.append("")
        for idx in range(ARM_JOINT_COUNT):
            lines.append(f"{JOINT_NAMES[idx]:>12}: {joint_state[idx]:6.3f} rad")

        for idx, text in enumerate(lines):
            surface = font.render(text, True, (230, 230, 230))
            screen.blit(surface, (12, 10 + idx * 20))
        pygame.display.flip()

    while running:
        while conn.poll():
            msg = conn.recv()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "state":
                joint_state = np.asarray(
                    msg.get("joints", joint_state), dtype=np.float64
                )
                eef_state = np.asarray(msg.get("eef", eef_state), dtype=np.float64)
                gripper_closed = bool(msg.get("gripper", gripper_closed))
            elif msg.get("type") == "shutdown":
                running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                conn.send({"type": "quit"})
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    conn.send({"type": "quit"})
                    running = False
                elif event.key == pygame.K_SPACE:
                    conn.send({"type": "toggle_gripper"})
                else:
                    pressed.add(event.key)
                    conn.send({"type": "keys", "keys": list(pressed)})
            elif event.type == pygame.KEYUP:
                if event.key in pressed:
                    pressed.remove(event.key)
                    conn.send({"type": "keys", "keys": list(pressed)})

        draw_overlay()
        clock.tick(60)

    pygame.quit()
    conn.close()


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    parent_conn, child_conn = Pipe()
    input_proc = Process(target=pygame_loop, args=(child_conn,), daemon=True)
    input_proc.start()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    maybe_reset_keyframe(model, data)

    joint_limits = build_joint_limits(model, ARM_JOINT_COUNT)
    joint_targets = data.qpos[: ARM_JOINT_COUNT].copy()
    jaw_target = data.qpos[ARM_JOINT_COUNT]
    gripper_closed = jaw_target <= (JAW_OPEN + JAW_CLOSED) * 0.5
    eef_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY_NAME)
    if eef_body_id < 0:
        raise ValueError(f"Unable to find end-effector body '{EEF_BODY_NAME}' in the model.")
    mujoco.mj_forward(model, data)
    eef_target = data.xpos[eef_body_id].copy()
    ik_scratch = mujoco.MjData(model)
    ik_warning_printed = False
    active_keys: Set[int] = set()

    with viewer.launch_passive(model, data) as gui:
        print("Keyboard teleop ready. Focus the pygame window for key input.")
        running = True
        last_time = time.time()
        state_push_time = 0.0
        while running and gui.is_running():
            now = time.time()
            dt = max(1.0 / args.max_hz, now - last_time)
            last_time = now

            while parent_conn.poll():
                msg = parent_conn.recv()
                if not isinstance(msg, dict):
                    continue
                msg_type = msg.get("type")
                if msg_type == "keys":
                    active_keys = set(msg.get("keys", []))
                elif msg_type == "toggle_gripper":
                    gripper_closed = not gripper_closed
                    jaw_target = JAW_CLOSED if gripper_closed else JAW_OPEN
                elif msg_type == "quit":
                    running = False

            speed = args.linear_speed
            if pygame.K_LSHIFT in active_keys or pygame.K_RSHIFT in active_keys:
                speed *= args.fast_multiplier

            cart_delta = np.zeros(3, dtype=np.float64)
            for key, direction in CARTESIAN_BINDINGS.items():
                if key in active_keys:
                    cart_delta += direction

            if np.linalg.norm(cart_delta) > 1e-9:
                cart_delta = cart_delta / np.linalg.norm(cart_delta)
                desired_target = eef_target + cart_delta * speed * dt
                solution, achieved = solve_position_ik(
                    model,
                    data,
                    ik_scratch,
                    joint_targets,
                    desired_target,
                    joint_limits,
                    eef_body_id,
                    max_iters=args.ik_iters,
                    tol=5e-4,
                    damping=args.ik_damping,
                )
                if solution is not None:
                    joint_targets = solution
                    eef_target = desired_target
                    ik_warning_printed = False
                else:
                    if not ik_warning_printed:
                        print("IK failed to reach target; staying at previous pose.")
                        ik_warning_printed = True
                    eef_target = achieved

            for idx, (low, high) in enumerate(joint_limits):
                if high > low:
                    joint_targets[idx] = np.clip(joint_targets[idx], low, high)

            data.ctrl[: ARM_JOINT_COUNT] = joint_targets
            data.ctrl[ARM_JOINT_COUNT] = jaw_target

            mujoco.mj_step(model, data)
            gui.sync()

            now = time.time()
            if now - state_push_time > 1.0 / 15.0:
                try:
                    parent_conn.send(
                        {
                            "type": "state",
                            "joints": joint_targets.tolist(),
                            "gripper": gripper_closed,
                            "eef": eef_target.tolist(),
                        }
                    )
                    state_push_time = now
                except (BrokenPipeError, EOFError):
                    running = False

    try:
        parent_conn.send({"type": "shutdown"})
    except (BrokenPipeError, EOFError):
        pass
    parent_conn.close()
    input_proc.join(timeout=1.0)


if __name__ == "__main__":
    main()

