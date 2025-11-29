"""MuJoCo pick-and-place environment for the trs_so_arm100 (SO100 arm).

Task: Pick the red cube and place it 5 cm to the left (-X direction).
5 continuous arm position actions + 1 binary gripper action (0=open, 1=close).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPO_ROOT / "mujoco_sim" / "trs_so_arm100" / "scene.xml"
TARGET_OFFSET = np.array([-0.05, 0.0, 0.0], dtype=np.float32)  # 5 cm left


class RedCubePickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        render_mode: Optional[str] = None,
        frame_skip: int = 3,
        max_episode_steps: int = 600,
    ) -> None:
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer = None

        scene_path = Path(model_path or DEFAULT_SCENE_PATH).expanduser().resolve()

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * self.frame_skip

        assert self.model.nu == 6, f"Expected 6 actuators, got {self.model.nu}"
        self.n_arm = 5
        self.n_gripper = 1

        # Control limits (from actuator definition in XML)
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # Action space: Flattened continuous (5 arm + 1 gripper)
        # Gripper: [-1, 0] -> open, (0, 1] -> close
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_arm + 1,), dtype=np.float32
        )

        # Observation space
        obs_dim = 2 * 6 + 3 + 3 + 3  # qpos(6), qvel(6), cube(3), ee(3), target(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # IDs
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw")
        self.home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home_with_cube")

        # Cube freejoint slices
        jnt_adr = self.model.body_jntadr[self.cube_body_id]
        self.cube_qpos_adr = self.model.jnt_qposadr[jnt_adr]
        self.cube_qvel_adr = self.model.jnt_dofadr[jnt_adr]

        self._cube_spawn_pos: Optional[np.ndarray] = None
        self._target_pos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps = 0
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        self._cube_spawn_pos = cube_pos.copy()
        self._target_pos = cube_pos + TARGET_OFFSET

        return self._get_obs(), {
            "target_position": self._target_pos.copy(),
            "is_success": False,
        }

    def step(self, action: np.ndarray):
        # Expect action shape (6,) -> 5 arm + 1 gripper
        action = np.asarray(action, dtype=np.float32).flatten()
        
        arm_action = np.clip(action[:self.n_arm], -1.0, 1.0)
        gripper_raw = action[-1]
        
        # Threshold gripper: > 0.0 -> Close (1), <= 0.0 -> Open (0)
        gripper_cmd = 1 if gripper_raw > 0.0 else 0

        # Map normalized arm action [-1, 1] → joint position range
        target_arm_pos = (
            0.5 * (self.ctrl_low[:self.n_arm] + self.ctrl_high[:self.n_arm]) +
            0.5 * (self.ctrl_high[:self.n_arm] - self.ctrl_low[:self.n_arm]) * arm_action
        )

        # Binary gripper → open or closed position
        target_gripper_pos = (
            self.ctrl_high[-1] if gripper_cmd == 1 else self.ctrl_low[-1]
        )

        # Apply control
        self.data.ctrl[:self.n_arm] = target_arm_pos
        self.data.ctrl[-1] = target_gripper_pos

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1

        obs = self._get_obs()
        reward, reward_info, success = self._compute_reward(arm_action, gripper_cmd)
        terminated = success
        truncated = self._elapsed_steps >= self._max_episode_steps
        info = {
            "is_success": success,
            "target_position": self._target_pos.copy(),
            "cube_position": self.data.xpos[self.cube_body_id].copy(),
            "ee_position": self.data.xpos[self.ee_body_id].copy(),
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[:6],           # 6 joints (5 arm + 1 gripper)
            self.data.qvel[:6],
            self.data.xpos[self.cube_body_id],
            self.data.xpos[self.ee_body_id],
            self._target_pos,
        ]).astype(np.float32)

    def _compute_reward(self, arm_action: np.ndarray, gripper_cmd: int):
        cube_pos = self.data.xpos[self.cube_body_id]
        ee_pos = self.data.xpos[self.ee_body_id]
        target_pos = self._target_pos

        dist_ee_to_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_cube_to_target = np.linalg.norm(cube_pos - target_pos)
        is_lifted = cube_pos[2] > 0.05
        is_placed = dist_cube_to_target < 0.03 and is_lifted

        # Reach reward
        reach_reward = 1.0 - np.tanh(5.0 * dist_ee_to_cube)

        # Control penalty
        ctrl_penalty = -0.01 * np.sum(np.square(arm_action))

        # Gripper shaping
        proximity = np.clip(1.0 - 5.0 * dist_ee_to_cube, 0.0, 1.0)
        gripper_open = 1.0 if gripper_cmd == 0 else 0.0
        gripper_closed = 1.0 - gripper_open

        r_open_when_far = gripper_open * (1.0 - proximity)
        r_close_when_near = gripper_closed * proximity
        gripper_shaping = 0.5 * (r_open_when_far + r_close_when_near)

        # Bonuses
        grasp_bonus = 2.0 if is_lifted else 0.0
        place_bonus = 10.0 if is_placed else 0.0

        total = reach_reward + ctrl_penalty + gripper_shaping + grasp_bonus + place_bonus
        success = bool(is_placed)

        info = {
            "reward_reach": float(reach_reward),
            "reward_ctrl": float(ctrl_penalty),
            "reward_gripper_shaping": float(gripper_shaping),
            "reward_grasp_bonus": float(grasp_bonus),
            "reward_place_bonus": float(place_bonus),
            "dist_ee_cube": float(dist_ee_to_cube),
            "dist_cube_target": float(dist_cube_to_target),
        }

        return total, info, success

    def current_joint_positions(self) -> np.ndarray:
        """Return the controllable joint positions (useful for logging)."""
        return self.data.qpos[: self.n_arm].copy()

    def render(self):
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        # rgb_array
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)  # <--- No camera_id argument
        return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


__all__ = ["RedCubePickEnv"]