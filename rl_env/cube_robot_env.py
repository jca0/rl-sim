"""MuJoCo pick-and-place environment for the trs_so_arm100 scene.

The task: pick the red cube at its fixed spawn pose and place it 5 cm to the
left (negative X offset in the MuJoCo world frame). This environment exposes
joint-space position control, shaped rewards, and RGB rendering suitable for
training with SAC or PPO via Stable Baselines3.
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
TARGET_OFFSET = np.array([-0.05, 0.0, 0.0], dtype=np.float32)  # 5 cm to the left


class RedCubePickEnv(gym.Env):
    """Gymnasium-compatible MuJoCo environment for the red-cube pick/place task."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 600,  # Changed to 600 (6 seconds)
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{render_mode}'")

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._renderer: Optional[mujoco.Renderer] = None

        scene_path = Path(model_path or DEFAULT_SCENE_PATH).expanduser().resolve()
        if not scene_path.exists():
            raise FileNotFoundError(f"MuJoCo scene not found at {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * self.frame_skip

        self._arm_dofs = self.model.nu
        if self._arm_dofs == 0:
            raise ValueError("Scene defines no actuators; cannot build control env")

        # Pre-compute control ranges (position actuators inherit joint ranges).
        ctrl_range = self.model.actuator_ctrlrange.copy()
        print(f"ctrl_range: {ctrl_range}")
        self._ctrl_center = ctrl_range.mean(axis=1)
        self._ctrl_half_range = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])

        # Gripper is the last actuator
        self._gripper_ctrl_range = ctrl_range[-1]

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._arm_dofs,), dtype=np.float32
        )

        obs_dim = 2 * self._arm_dofs + 3 + 3 + 3  # qpos, qvel, cube xyz, ee xyz, target xyz
        obs_low = -np.ones(obs_dim, dtype=np.float32) * np.inf
        obs_high = np.ones(obs_dim, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block"
        )
        self._ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw"
        )
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home_with_cube"
        )

        # Locate the cube's freejoint slices for direct qpos/qvel edits.
        cube_jnt_adr = self.model.body_jntadr[self._cube_body_id]
        if cube_jnt_adr < 0:
            raise ValueError("Cube body does not have a joint; expected a freejoint.")
        self._cube_qpos_slice = slice(
            self.model.jnt_qposadr[cube_jnt_adr],
            self.model.jnt_qposadr[cube_jnt_adr] + 7,
        )
        self._cube_qvel_slice = slice(
            self.model.jnt_dofadr[cube_jnt_adr],
            self.model.jnt_dofadr[cube_jnt_adr] + 6,
        )

        self._cube_spawn_pos: Optional[np.ndarray] = None
        self._target_pos: Optional[np.ndarray] = None

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ Gym API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps = 0
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        self._cube_spawn_pos = cube_pos
        self._target_pos = cube_pos + TARGET_OFFSET

        observation = self._get_obs()
        info = {
            "target_position": self._target_pos.copy(),
            "cube_position": cube_pos,
        }

        if self.render_mode == "rgb_array":
            self.render()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self._arm_dofs,):
            raise ValueError(f"Action must be shape {(self._arm_dofs,)}, got {action.shape}")

        # 1. Arm control (indices 0..N-2): Map [-1, 1] -> [min, max]
        # 2. Gripper control (index N-1): Binary 0 (open) or 1 (closed)
        #    User input > 0.5 is interpreted as closed.
        
        # Split action
        arm_action = action[:-1]
        gripper_action_val = action[-1]

        # Compute arm targets
        bounded_arm = np.clip(arm_action, -1.0, 1.0)
        target_arm_ctrl = (
            self._ctrl_center[:-1] + bounded_arm * self._ctrl_half_range[:-1]
        )

        # Compute gripper target
        # 0 -> Open (min range), 1 -> Closed (max range)
        if gripper_action_val > 0.5:
            target_gripper_ctrl = self._gripper_ctrl_range[1]  # Closed (max)
        else:
            target_gripper_ctrl = self._gripper_ctrl_range[0]  # Open (min)

        # Apply to simulation
        self.data.ctrl[:-1] = target_arm_ctrl
        self.data.ctrl[-1] = target_gripper_ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1

        # Pass action to compute_reward for regularization
        total_reward, reward_info, success = self._compute_reward(action)
        observation = self._get_obs()
        terminated = success
        truncated = self._elapsed_steps >= self._max_episode_steps
        info = {
            "reward_terms": reward_info,
            "cube_position": self.data.xpos[self._cube_body_id].copy(),
            "ee_position": self.data.xpos[self._ee_body_id].copy(),
            "target_position": self._target_pos.copy(),
            "is_success": success,
        }

        if self.render_mode == "rgb_array":
            self.render()

        return observation, total_reward, terminated, truncated, info

    # ---------------------------------------------------------------- Utilities
    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[: self._arm_dofs]
        joint_vel = self.data.qvel[: self._arm_dofs]
        cube_pos = self.data.xpos[self._cube_body_id]
        ee_pos = self.data.xpos[self._ee_body_id]
        target_pos = self._target_pos if self._target_pos is not None else cube_pos

        return np.concatenate(
            [
                joint_pos,
                joint_vel,
                cube_pos,
                ee_pos,
                target_pos,
            ]
        ).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float], bool]:
        assert self._cube_spawn_pos is not None
        
        cube_pos = self.data.xpos[self._cube_body_id]
        ee_pos = self.data.xpos[self._ee_body_id]

        # Distance definitions
        d_reach = np.linalg.norm(ee_pos - cube_pos)
        
        # Check grasp status (Logic: is cube lifted off table?)
        # Threshold height 0.05 (table is at 0 in simulation world usually, but here cube starts at 0.03)
        is_grasped = cube_pos[2] > 0.05

        # 1. Reach Reward
        r_reach = 1 - np.tanh(10.0 * d_reach)
        
        # Action penalty
        r_ctrl = -0.01 * np.square(action).sum()

        # Gripper shaping
        # Identify gripper range: index 0 is Open, index 1 is Closed
        min_g, max_g = self._gripper_ctrl_range
        jaw_pos = self.data.qpos[self._arm_dofs - 1]
        
        # Normalize to [0 (open), 1 (closed)]
        # We clamp to handle small numerical violations
        jaw_range = max_g - min_g
        if jaw_range > 1e-8:
            norm_jaw = np.clip((jaw_pos - min_g) / jaw_range, 0.0, 1.0)
        else:
            norm_jaw = 0.0
        
        openness = 1.0 - norm_jaw
        closedness = norm_jaw
        proximity = 1.0 - np.tanh(10.0 * d_reach)
        
        # 1. Descend with open gripper: Reward openness when NOT close
        # Weighted by (1 - proximity) so it applies when far
        r_open_shaping = openness * (1.0 - proximity)
        
        # 2. Grasp: Reward closedness when close
        # Weighted by proximity so it applies when close
        r_close_shaping = closedness * proximity
        
        # Combined shaping (weight 0.5 to not overpower reach)
        r_gripper_shaping = 0.5 * (r_open_shaping + r_close_shaping)

        # Composition
        grasp_bonus_val = 0.0
        success_bonus_val = 0.0
        success = is_grasped

        if success:
            grasp_bonus_val = 2.0
            success_bonus_val = 10.0
        
        total_reward = r_reach + r_ctrl + r_gripper_shaping + grasp_bonus_val + success_bonus_val
        
        reward_info = {
            "reach": r_reach,
            "ctrl": r_ctrl,
            "gripper_shaping": r_gripper_shaping,
            "grasp_bonus": grasp_bonus_val,
            "success_bonus": success_bonus_val,
        }

        return total_reward, reward_info, success

    # ------------------------------------------------------------ Helper hooks
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            if hasattr(self._renderer, "close"):
                self._renderer.close()
            elif hasattr(self._renderer, "free"):
                self._renderer.free()
            self._renderer = None

    # ------------------------------------------------------ Logging interfaces
    def current_joint_positions(self) -> np.ndarray:
        """Return the controllable joint positions (useful for logging)."""
        return self.data.qpos[: self._arm_dofs].copy()

    def current_joint_velocities(self) -> np.ndarray:
        return self.data.qvel[: self._arm_dofs].copy()


__all__ = ["RedCubePickEnv"]
