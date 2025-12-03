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
        
        # INCREASE GRIPPER GAINS FOR STABILITY
        # Find the actuator for the jaw
        for i in range(self.model.nu):
            if "Jaw" in self.model.actuator(i).name:
                # Boost Kp from 50 to 500
                self.model.actuator_gainprm[i, 0] = 500.0
                self.model.actuator_biasprm[i, 1] = -500.0
                # Boost Force Range from 3.5 to 10.0
                self.model.actuator_ctrlrange[i, :] = [-0.174, 1.75] # This is pos range
                # To boost force, we usually edit actuator_forcerange if it exists, 
                # but position actuators usually rely on gain/bias.
                # Let's just boost the KP.
        
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
        self.fixed_jaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.fixed_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_1")
        self.moving_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_1")

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
        gripper_action = np.clip(action[-1], -1.0, 1.0)

        # Map normalized arm action [-1, 1] → joint position range
        target_arm_pos = (
            0.5 * (self.ctrl_low[:self.n_arm] + self.ctrl_high[:self.n_arm]) +
            0.5 * (self.ctrl_high[:self.n_arm] - self.ctrl_low[:self.n_arm]) * arm_action
        )

        # Map normalized gripper action [-1, 1] → gripper position range (CONTINUOUS!)
        target_gripper_pos = (
            0.5 * (self.ctrl_low[-1] + self.ctrl_high[-1]) +
            0.5 * (self.ctrl_high[-1] - self.ctrl_low[-1]) * gripper_action
        )
        
        # For reward computation, determine if gripper is "closing" (for compatibility)
        gripper_cmd = 1 if gripper_action > 0.0 else 0

        # Apply control
        self.data.ctrl[:self.n_arm] = target_arm_pos
        self.data.ctrl[-1] = target_gripper_pos

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(arm_action, gripper_cmd)
        
        # Define success: Cube lifted > 5 cm (0.05m)
        # Note: Cube center starts at ~0.025m
        cube_height = self.data.xpos[self.cube_body_id][2]
        is_lifted = cube_height > 0.05
        success = is_lifted
        
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
        
        GRASP_RADIUS = 0.02
        TARGET_HEIGHT = 0.1

        # STEP 1: Reach towards cube
        reward_reach = 1.0 - np.tanh(10.0 * dist_ee_to_cube)

        # STEP 2: Grasp reward (simplified)
        # Reward simply for having the gripper closed near the object
        # We trust the physics: if we close near object and lift, it works
        is_near_object = dist_ee_to_cube < 0.03
        
        if is_near_object and gripper_cmd == 1:
            reward_grasp = 1.0
        else:
            reward_grasp = 0.0

        # STEP 3: Lift cube
        # This is the most important part
        if cube_pos[2] > 0.05: # Lifted > 5cm
            reward_lift = 5.0
        elif cube_pos[2] > 0.03: # Lifted slightly (start of lift)
            reward_lift = 1.0
        else:
            reward_lift = 0.0

        # penalize high arm action
        reward_ctrl = -np.sum(np.square(arm_action)) * 0.1
        
        # Penalize high velocity of the cube (instability/flying away)
        cube_vel = self.data.qvel[self.cube_qvel_adr : self.cube_qvel_adr + 3]
        reward_cube_vel = -np.linalg.norm(cube_vel) * 0.07

        # Orient gripper downwards
        fixed_jaw_mat = self.data.xmat[self.fixed_jaw_id].reshape(3, 3)
        local_y_global = fixed_jaw_mat[:, 1]  # Global direction of local Y
        reward_orientation = np.dot(local_y_global, np.array([0, 0, 1.0], dtype=np.float32)) 
        reward_orientation = max(0.0, reward_orientation) * 0.1

        # STEP 4: Place reward (GATED by lift)
        # Only reward moving to target if we have lifted the object
        # This prevents "bulldozing" (pushing along the ground)
        if cube_pos[2] > 0.03: # Only if lifted > 3cm
            dist_cube_to_target = np.linalg.norm(cube_pos - target_pos)
            reward_place = 1.0 - np.tanh(5.0 * dist_cube_to_target)
            reward_place *= 5.0 # Big bonus for placing
        else:
            reward_place = 0.0

        total_reward = reward_reach + reward_grasp + reward_lift + reward_place + reward_ctrl + reward_cube_vel + reward_orientation


        info = {
            "reward_reach": float(reward_reach),
            "reward_grasp": float(reward_grasp),
            "reward_lift": float(reward_lift),
            "reward_place": float(reward_place),
            "reward_ctrl": float(reward_ctrl),
            "reward_cube_vel": float(reward_cube_vel),
            "reward_orientation": float(reward_orientation),
            "dist_ee_cube": float(dist_ee_to_cube),
            "in_grasp_zone": dist_ee_to_cube < GRASP_RADIUS,
            "gripper_cmd": gripper_cmd
        }

        return total_reward, info

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