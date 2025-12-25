import mujoco
from mujoco.glfw import glfw
import numpy as np
import time

# --------------------
# Simple 2-joint arm
# --------------------
XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# --------------------
# GLFW setup
# --------------------
glfw.init()
window = glfw.create_window(800, 600, "MuJoCo Live", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# --------------------
# MuJoCo render objects
# --------------------
cam = mujoco.MjvCamera()
cam.distance = 1.0
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# --------------------
# Main loop
# --------------------
t = 0.0
while not glfw.window_should_close(window):
    # demo motion (we’ll replace this with MediaPipe)
    data.qpos[0] = np.sin(t)
    data.qpos[1] = np.cos(t)

    mujoco.mj_step(model, data)
    mujoco.mjv_updateScene(
        model, data, opt, None, cam,
        mujoco.mjtCatBit.mjCAT_ALL, scene
    )

    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

    t += 0.02
    time.sleep(0.01)

glfw.terminate()
