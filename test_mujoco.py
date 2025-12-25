import mujoco
from mujoco.glfw import glfw
import time

xml = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="sphere" size="0.02"/>
      <body name="link1">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
        <body name="link2" pos="0.1 0 0">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# GLFW
glfw.init()
window = glfw.create_window(640, 480, "MuJoCo Render", None, None)
glfw.make_context_current(window)

# Scene
scene = mujoco.MjvScene(model, maxgeom=1000)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()

cam.distance = 0.5
cam.azimuth = 45
cam.elevation = -30

# Renderer (NEW API)
ctx = mujoco.MjrContext()
mujoco.mjr_makeContext(
    model,
    ctx,
    mujoco.mjtFontScale.mjFONTSCALE_150
)

while not glfw.window_should_close(window):
    data.qpos[0] = 0.5 * time.time()
    data.qpos[1] = 0.25 * time.time()

    mujoco.mj_step(model, data)
    mujoco.mjv_updateScene(model, data, opt, None, cam, 0, scene)

    w, h = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()
    time.sleep(0.016)

glfw.terminate()
