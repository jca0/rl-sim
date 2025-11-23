import mujoco
from mujoco import MjModel, MjData, mj_resetDataKeyframe
import mujoco.viewer as viewer

model = MjModel.from_xml_path("trs_so_arm100/scene.xml")
data = MjData(model)


def reset_to_keyframe(name: str) -> bool:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id >= 0:
        mj_resetDataKeyframe(model, data, key_id)
        return True
    return False

reset_to_keyframe("rest_with_cube")

with viewer.launch_passive(model, data) as gui:
    while gui.is_running():
        mujoco.mj_step(model, data)
        gui.sync()