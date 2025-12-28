import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("simple_arm.xml")
data = mujoco.MjData(model)

shoulder_id = model.joint("shoulder").qposadr
elbow_id = model.joint("elbow").qposadr

t = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.qpos[shoulder_id] = 0.6 * np.sin(t)
        data.qpos[elbow_id] = 0.8 * np.sin(t + 1.0)

        mujoco.mj_step(model, data)
        viewer.sync()

        t += 0.02
        time.sleep(0.01)
