from gymnasium.envs.registration import register

# Expose the environment class for consumers who prefer direct imports.
from .red_cube_pick_env import RedCubePickEnv  # noqa: F401


def _register_env() -> None:
    """Register the environment with Gymnasium only once."""
    try:
        register(
            id="RedCubePick-v0",
            entry_point="rl_env.red_cube_pick_env:RedCubePickEnv",
            max_episode_steps=200,
        )
    except Exception:
        # Gymnasium raises an Error if the env is registered twice; swallow it.
        pass


_register_env()

