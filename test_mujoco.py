import time
import mujoco
from mujoco import MjModel, MjData, mj_resetDataKeyframe
import mujoco.viewer as viewer

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

model = MjModel.from_xml_path("trs_so_arm100/scene.xml")
data = MjData(model)


def reset_to_keyframe(name: str) -> bool:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id >= 0:
        mj_resetDataKeyframe(model, data, key_id)
        return True
    return False


def joint_qpos_snapshot() -> dict[str, float]:
    """Return current joint positions keyed by joint name."""
    return {name: float(data.qpos[idx]) for idx, name in enumerate(JOINT_NAMES)}


reset_to_keyframe("rest_with_cube")

with viewer.launch_passive(model, data) as gui:
    start = time.time()
    last_log = start
    while gui.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)