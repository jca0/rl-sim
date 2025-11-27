"""
SO-100 pick-and-place Gymnasium environment for PPO training.

This environment loads the full MuJoCo scene (arm + cube) and exposes a
position-control interface over the six actuated joints. The agent must pick up
the cube resting on the table and place it 10 cm to the robot's left.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


@dataclass
class RewardWeights:
    reach: float = 2.0
    lift: float = 2.5
    place: float = 4.0
    grasp_bonus: float = 1.0
    success_bonus: float = 10.0
    action_penalty: float = 0.01
    time_penalty: float = 0.005


class SOPickPlaceEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Pick-and-place task for the SO-100 robot arm.

    Actions represent normalized deltas in joint position goals (-1..1). The
    environment internally clips goals to the physical joint ranges and applies
    a MuJoCo position actuator on each joint.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_xml_path: Optional[str] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 250,
        target_offset: Tuple[float, float, float] = (0.0, 0.10, 0.0),
        cube_xy_noise: float = 0.01,
        target_jitter: float = 0.01,
        lift_height: float = 0.07,
        success_tolerance: float = 0.025,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_model = os.path.join(base_dir, "trs_so_arm100", "scene.xml")
        self.model_xml_path = model_xml_path or default_model

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.target_offset = np.asarray(target_offset, dtype=np.float64)
        self.cube_xy_noise = cube_xy_noise
        self.target_jitter = target_jitter
        self.lift_height = lift_height
        self.success_tolerance = success_tolerance
        self.render_mode = render_mode

        self.reward_weights = RewardWeights(**(reward_weights or {}))
        self.np_random, _ = seeding.np_random(seed)

        self.model = mujoco.MjModel.from_xml_path(self.model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.sim_timestep = self.model.opt.timestep
        self.dt = self.sim_timestep * self.frame_skip

        self.rest_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "rest_with_cube"
        )
        if self.rest_key_id < 0:
            raise RuntimeError("Keyframe 'rest_with_cube' is required in the scene xml.")

        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block"
        )
        if self.cube_body_id < 0:
            raise RuntimeError("Body named 'block' (the cube) is missing from the scene.")

        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw"
        )
        if self.ee_body_id < 0:
            raise RuntimeError("Body 'Moving_Jaw' not found; end-effector pose unavailable.")

        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        if any(jid < 0 for jid in joint_ids):
            raise RuntimeError("Failed to find all joint ids for control.")
        self.joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self.ctrl_dim = len(self.joint_ids)
        self.ctrl_lower = self.model.jnt_range[self.joint_ids, 0]
        self.ctrl_upper = self.model.jnt_range[self.joint_ids, 1]
        ranges = self.ctrl_upper - self.ctrl_lower
        self.action_scale = 0.02 * ranges
        self.action_scale[-1] = 0.05  # jaw can move a bit faster

        cube_joint_id = self.model.body_jntadr[self.cube_body_id]
        self.cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.cube_qvel_addr = self.model.jnt_dofadr[cube_joint_id]
        self.cube_pos_slice = slice(self.cube_qpos_addr, self.cube_qpos_addr + 3)
        self.cube_vel_slice = slice(self.cube_qvel_addr, self.cube_qvel_addr + 3)

        self.renderer = None
        self._reset_to_keyframe()
        self.default_cube_qpos = self.data.qpos[
            self.cube_qpos_addr : self.cube_qpos_addr + 7
        ].copy()
        self.ctrl_target = self.data.qpos[: self.ctrl_dim].copy()
        self.data.ctrl[:] = self.ctrl_target
        self.step_count = 0
        self.last_action = np.zeros(self.ctrl_dim, dtype=np.float64)
        self.target_pos = np.zeros(3, dtype=np.float64)

        sample_obs = self._get_obs()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.ctrl_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )


    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.step_count = 0
        self._reset_to_keyframe()

        self.ctrl_target = self.data.qpos[: self.ctrl_dim].copy()
        self.data.ctrl[:] = self.ctrl_target
        self.last_action[:] = 0.0

        # Randomize cube XY within a small patch on the table surface.
        cube_xy_noise = self.np_random.uniform(-self.cube_xy_noise, self.cube_xy_noise, size=2)
        new_cube_pos = self.default_cube_qpos[:3].copy()
        new_cube_pos[:2] += cube_xy_noise
        self.data.qpos[self.cube_pos_slice] = new_cube_pos
        self.data.qvel[self.cube_vel_slice] = 0.0

        # Target is 10 cm to the robot's left (+y) plus small jitter.
        jitter = self.np_random.uniform(-self.target_jitter, self.target_jitter, size=3)
        jitter[2] = 0.0  # keep height identical
        self.target_pos = new_cube_pos + self.target_offset + jitter

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = {"target_position": self.target_pos.copy()}
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        action_delta = action * self.action_scale
        self.ctrl_target = np.clip(
            self.ctrl_target + action_delta, self.ctrl_lower, self.ctrl_upper
        )
        self.data.ctrl[:] = self.ctrl_target
        self.last_action = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, metrics = self._compute_reward()

        terminated = bool(metrics["success"])
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "reach_distance": metrics["reach_distance"],
            "place_distance": metrics["place_distance"],
            "cube_height": metrics["cube_height"],
            "success": float(metrics["success"]),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _reset_to_keyframe(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.rest_key_id)
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        joint_qpos = self.data.qpos[: self.ctrl_dim]
        joint_qvel = self.data.qvel[: self.ctrl_dim]
        cube_pos = self.data.qpos[self.cube_pos_slice]
        cube_vel = self.data.qvel[self.cube_vel_slice]
        ee_pos = self.data.xpos[self.ee_body_id]

        obs = np.concatenate(
            [
                joint_qpos,
                joint_qvel,
                self.ctrl_target,
                cube_pos,
                cube_vel,
                ee_pos,
                self.target_pos,
                ee_pos - cube_pos,
                cube_pos - self.target_pos,
                [joint_qpos[-1]],  # jaw opening
                [cube_pos[2]],
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        ee_pos = self.data.xpos[self.ee_body_id]
        cube_pos = self.data.qpos[self.cube_pos_slice]
        cube_height = float(cube_pos[2])

        reach_distance = float(np.linalg.norm(ee_pos - cube_pos))
        place_distance = float(np.linalg.norm(cube_pos - self.target_pos))

        reach_reward = self.reward_weights.reach * (1.0 - np.tanh(3.0 * reach_distance))
        place_reward = self.reward_weights.place * (1.0 - np.tanh(3.0 * place_distance))
        lift_reward = self.reward_weights.lift * float(cube_height > self.lift_height)
        grasp_reward = self.reward_weights.grasp_bonus * float(
            cube_height > self.lift_height and self.ctrl_target[-1] < 0.4
        )

        success = bool(
            cube_height > (self.lift_height - 0.01) and place_distance < self.success_tolerance
        )
        success_bonus = self.reward_weights.success_bonus * float(success)

        action_penalty = self.reward_weights.action_penalty * float(
            np.mean(np.square(self.last_action))
        )
        time_penalty = self.reward_weights.time_penalty

        reward = (
            reach_reward
            + lift_reward
            + place_reward
            + grasp_reward
            + success_bonus
            - action_penalty
            - time_penalty
        )

        return reward, {
            "reach_distance": reach_distance,
            "place_distance": place_distance,
            "cube_height": cube_height,
            "success": success,
        }


def make_so_pick_place_env(**kwargs) -> SOPickPlaceEnv:
    """Convenience helper mirroring gym.make-style construction."""
    return SOPickPlaceEnv(**kwargs)

