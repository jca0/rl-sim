"""
Parameterized SO-100 MuJoCo Environment for Sim2Real Learning.

This environment wraps the SO-100 robot in MuJoCo with tunable simulation parameters
for system identification and parameter optimization.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any
import mujoco


class ParameterizedSO100Env(gym.Env):
    """
    Gymnasium environment for SO-100 robot with parameterizable physics.
    
    This environment allows tuning of simulation parameters like friction,
    damping, actuator gains, etc. to match real-world behavior.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    # Default simulation parameters (nominal)
    DEFAULT_PARAMS = {
        # Actuator parameters
        "kp": 100.0,           # Position gain
        "kv": 10.0,            # Velocity gain
        "control_delay": 0.0,  # Control delay (seconds)
        
        # Joint friction/damping
        "joint_damping": 0.5,
        "joint_friction": 0.1,
        
        # Contact parameters  
        "robot_friction": 1.0,  # Friction coefficient for robot links
        "table_friction": 0.8,  # Table friction
        "cube_friction": 0.6,   # Cube friction
        
        # Object properties
        "cube_mass": 0.05,      # Cube mass (kg)
    }
    
    def __init__(
        self,
        urdf_path: Optional[str] = None,
        task: str = "reach",
        render_mode: Optional[str] = None,
        control_freq: int = 50,
        simulation_params: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the SO-100 environment.
        
        Args:
            urdf_path: Path to SO-100 URDF file
            task: Task type ("reach", "pick", "place")
            render_mode: Rendering mode
            control_freq: Control frequency (Hz)
            simulation_params: Custom simulation parameters
        """
        super().__init__()
        
        self.task = task
        self.render_mode = render_mode
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        
        # Simulation parameters (start with defaults)
        self.sim_params = self.DEFAULT_PARAMS.copy()
        if simulation_params:
            self.sim_params.update(simulation_params)
        
        # Find URDF path if not provided
        if urdf_path is None:
            # Try to find it relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            urdf_path = os.path.join(base_dir, "..", "Simulation", "SO100", "so100.urdf")
        
        self.urdf_path = urdf_path
        
        # Load MuJoCo model
        self._load_model()
        
        # Define action and observation spaces
        # Action: 6 joint positions + 1 gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # Observation: joint positions (7) + velocities (7) + cube pose (7)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_episode_steps = 200
        
    def _load_model(self):
        """Load or reload MuJoCo model with current parameters."""
        # For now, create a simple model
        # Your teammate will integrate the actual URDF
        
        xml_string = self._generate_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        
        # Apply simulation parameters
        self._apply_sim_params()
        
    def _generate_xml(self) -> str:
        """Generate MuJoCo XML with current parameters."""
        # Simplified XML - your teammate should replace with actual URDF loading
        xml = f"""
        <mujoco model="so100">
            <compiler angle="radian" />
            <option timestep="{self.dt}" gravity="0 0 -9.81" />
            
            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" />
                <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1" friction="{self.sim_params['table_friction']} 0.005 0.0001"/>
                
                <!-- Simple robot arm placeholder -->
                <body name="base" pos="0 0 0.1">
                    <geom type="cylinder" size="0.05 0.05" rgba="0.7 0.7 0.7 1" friction="{self.sim_params['robot_friction']} 0.005 0.0001"/>
                    <joint name="shoulder_pan" type="hinge" axis="0 0 1" damping="{self.sim_params['joint_damping']}" frictionloss="{self.sim_params['joint_friction']}"/>
                    
                    <body name="link1" pos="0 0 0.1">
                        <geom type="box" size="0.04 0.04 0.1" rgba="0.2 0.2 0.8 1" friction="{self.sim_params['robot_friction']} 0.005 0.0001"/>
                        <joint name="shoulder_lift" type="hinge" axis="0 1 0" damping="{self.sim_params['joint_damping']}" frictionloss="{self.sim_params['joint_friction']}"/>
                        
                        <body name="link2" pos="0 0 0.15">
                            <geom type="box" size="0.03 0.03 0.12" rgba="0.2 0.2 0.8 1" friction="{self.sim_params['robot_friction']} 0.005 0.0001"/>
                            <joint name="elbow_flex" type="hinge" axis="0 1 0" damping="{self.sim_params['joint_damping']}" frictionloss="{self.sim_params['joint_friction']}"/>
                        </body>
                    </body>
                </body>
                
                <!-- Target cube -->
                <body name="cube" pos="0.3 0.3 0.025">
                    <geom name="cube_geom" type="box" size="0.025 0.025 0.025" rgba="1 0 0 1" 
                          mass="{self.sim_params['cube_mass']}" 
                          friction="{self.sim_params['cube_friction']} 0.005 0.0001"/>
                    <freejoint/>
                </body>
            </worldbody>
            
            <actuator>
                <position name="act_shoulder_pan" joint="shoulder_pan" kp="{self.sim_params['kp']}"/>
                <position name="act_shoulder_lift" joint="shoulder_lift" kp="{self.sim_params['kp']}"/>
                <position name="act_elbow_flex" joint="elbow_flex" kp="{self.sim_params['kp']}"/>
            </actuator>
        </mujoco>
        """
        return xml
        
    def _apply_sim_params(self):
        """Apply current simulation parameters to the model."""
        # Update actuator gains
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = self.sim_params['kp']
            # Velocity gain is typically in gainprm[1] for position actuators
            
        # Control delay would be handled in step() by buffering actions
        
    def set_parameters(self, params: Dict[str, float]):
        """
        Update simulation parameters and reload model.
        
        Args:
            params: Dictionary of parameter updates
        """
        self.sim_params.update(params)
        self._load_model()
        
    def get_parameters(self) -> Dict[str, float]:
        """Get current simulation parameters."""
        return self.sim_params.copy()
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial state slightly
        if seed is not None:
            np.random.seed(seed)
        
        # Small random joint positions
        self.data.qpos[:] = np.random.uniform(-0.1, 0.1, size=self.data.qpos.shape)
        
        # Reset cube position (if exists)
        if "cube" in [self.model.body(i).name for i in range(self.model.nbody)]:
            cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            cube_qpos_addr = self.model.body_jntadr[cube_body_id]
            if cube_qpos_addr >= 0:
                # Randomize cube position
                self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [
                    np.random.uniform(0.2, 0.4),
                    np.random.uniform(-0.2, 0.2),
                    0.025
                ]
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = {}
        
        return observation, info
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Joint positions and velocities
        qpos = self.data.qpos[:7].copy()  # First 7 for robot
        qvel = self.data.qvel[:7].copy()
        
        # Cube pose (if exists)
        cube_pose = np.zeros(7)
        try:
            cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            cube_qpos_addr = self.model.body_jntadr[cube_body_id]
            if cube_qpos_addr >= 0:
                cube_pose = self.data.qpos[cube_qpos_addr:cube_qpos_addr+7].copy()
        except:
            pass
            
        obs = np.concatenate([qpos, qvel, cube_pose])
        return obs.astype(np.float32)
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Joint position targets (normalized -1 to 1)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Convert normalized action to joint positions
        # This is a placeholder - adjust based on actual joint limits
        action_scaled = action * np.pi  # Scale to radians
        
        # Apply control delay if specified
        if self.sim_params['control_delay'] > 0:
            # TODO: Implement action buffering for delay simulation
            pass
            
        # Set actuator controls
        self.data.ctrl[:] = action_scaled[:self.model.nu]
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_obs()
        
        # Compute reward based on task
        reward = self._compute_reward(action)
        
        # Check termination
        self.current_step += 1
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        info = {"step": self.current_step}
        
        return observation, reward, terminated, truncated, info
        
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward based on task type."""
        if self.task == "reach":
            # Reward for reaching target end-effector position
            # Placeholder - implement based on actual kinematics
            return -0.1  # Small penalty per step
            
        elif self.task == "pick":
            # Reward for picking up cube
            return 0.0
            
        elif self.task == "place":
            # Reward for placing cube at target
            return 0.0
            
        return 0.0
        
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Placeholder - implement task-specific termination
        return False
        
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # TODO: Implement rendering
            return np.zeros((480, 640, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            # TODO: Implement visual rendering
            pass
            
    def close(self):
        """Clean up resources."""
        pass


def main():
    """Test the environment."""
    print("Testing ParameterizedSO100Env...")
    
    # Create environment with custom parameters
    custom_params = {
        "kp": 150.0,
        "joint_damping": 1.0,
        "cube_mass": 0.1,
    }
    
    env = ParameterizedSO100Env(
        task="reach",
        simulation_params=custom_params
    )
    
    print("Environment created successfully")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Simulation parameters: {env.get_parameters()}")
    
    # Test reset and step
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    
    # Random actions
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={terminated or truncated}")
        
        if terminated or truncated:
            obs, info = env.reset()
            
    print("\n✓ Environment test complete")
    env.close()


if __name__ == "__main__":
    main()

