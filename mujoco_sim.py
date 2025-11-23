import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer as viewer

model = MjModel.from_xml_path("trs_so_arm100/scene.xml")
data = MjData(model)

with viewer.launch_passive(model, data):
    while True:
        mujoco.mj_step(model, data)