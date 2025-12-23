import cv2
import mediapipe as mp
import mujoco
import numpy as np
import time
import json

#config:
MODE = "replay"        # "record" or "replay"
OUTPUT_FILE = "motion.json"
FRAME_DELAY = 0.1      # seconds per frame (recording speed)

# MuJoCo: 2-joint planar arm

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


# Pose w/ MediaPipe

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = None
if MODE == "record":
    cap = cv2.VideoCapture(0)

# Motion Recording

recording = []
start_time = None

def record_frame(angle1, angle2):
    global start_time
    if start_time is None:
        start_time = time.time()

    t = time.time() - start_time
    recording.append({
        "t": round(t, 4),
        "joint1": float(angle1),
        "joint2": float(angle2)
    })

def save_recording():
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"frames": recording}, f, indent=2)
    print(f"[SAVED] {len(recording)} frames → {OUTPUT_FILE}")


# Kinematics (simple for now)

def compute_joint_angles(shoulder, elbow):
    vec = np.array(elbow) - np.array(shoulder)
    angle1 = np.arctan2(vec[1], vec[0])   # shoulder rotation
    angle2 = np.pi / 4                   # placeholder elbow
    return angle1, angle2


# MAIN

try:
    if MODE == "record":
        print("[MODE] RECORD — Ctrl+C to stop")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                shoulder = (lm[11].x, lm[11].y)  # left shoulder
                elbow = (lm[13].x, lm[13].y)     # left elbow

                angle1, angle2 = compute_joint_angles(shoulder, elbow)

                data.qpos[0] = angle1
                data.qpos[1] = angle2
                mujoco.mj_step(model, data)

                record_frame(angle1, angle2)

                print(f"[REC] joint1={angle1:.2f}, joint2={angle2:.2f}")

            time.sleep(FRAME_DELAY)

    elif MODE == "replay":
        print("[MODE] REPLAY")

        with open(OUTPUT_FILE) as f:
            motion = json.load(f)["frames"]

        start = time.time()

        for frame in motion:
            while time.time() - start < frame["t"]:
                time.sleep(0.001)

            data.qpos[0] = frame["joint1"]
            data.qpos[1] = frame["joint2"]
            mujoco.mj_step(model, data)

            print(f"[PLAY] joint1={frame['joint1']:.2f}, joint2={frame['joint2']:.2f}")

except KeyboardInterrupt:
    print("\n[EXIT]")

finally:
    if MODE == "record" and recording:
        save_recording()

    if cap:
        cap.release()
