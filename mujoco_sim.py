import time
import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer as viewer

model = MjModel.from_xml_path("trs_so_arm100/scene.xml")
data = MjData(model)

with viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Step the physics
        mujoco.mj_step(model, data)

        # Sync the viewer
        viewer.sync()

        # Rudimentary time keeping to match real-time
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
