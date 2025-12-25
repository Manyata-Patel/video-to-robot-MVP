import mujoco
from mujoco.glfw import glfw
import numpy as np
import time

xml = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="sphere" size="0.02" rgba="0.8 0.2 0.2 1"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" rgba="0.2 0.6 0.8 1"/>
        <body name="link2" pos="0.1 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" rgba="0.2 0.8 0.2 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

glfw.init()
window = glfw.create_window(640, 480, "MuJoCo Render", None, None)
glfw.make_context_current(window)

# Prepare scene
scn = mujoco.MjvScene(model, maxgeom=1000)
cam = mujoco.MjvCamera()
cam.distance = 1.0
opt = mujoco.MjvOption()

# Prepare renderer objects
con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)
    mujoco.mjv_updateScene(model, data, opt, None, cam, -1, scn)

    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(viewport, scn, con)

    glfw.swap_buffers(window)
    glfw.poll_events()
    time.sleep(0.016)

glfw.terminate()
