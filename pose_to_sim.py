import cv2
import numpy as np
import mediapipe as mp  # only this import for MediaPipe
import mujoco
import mujoco.viewer
import time

# ---------------------------
# MediaPipe setup
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# MuJoCo setup
# ---------------------------
model = mujoco.MjModel.from_xml_path("arm.xml")  # <-- your XML
data = mujoco.MjData(model)

shoulder_id = model.joint("shoulder").qposadr
elbow_id = model.joint("elbow").qposadr

# ---------------------------
# Angle helper
# ---------------------------
def angle_between(a, b, c):
    """
    Returns angle at point b (in radians)
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while cap.isOpened() and viewer.is_running():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Right arm landmarks
            shoulder = np.array([
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
            ])

            elbow = np.array([
                lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                lm[mp_pose.PoseLandmark.RIGHT_ELBOW].z
            ])

            wrist = np.array([
                lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                lm[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                lm[mp_pose.PoseLandmark.RIGHT_WRIST].z
            ])

            # Compute angles
            elbow_angle = angle_between(shoulder, elbow, wrist)
            shoulder_angle = np.arctan2(
                elbow[1] - shoulder[1],
                elbow[0] - shoulder[0]
            )

            # Apply to MuJoCo
            data.qpos[shoulder_id] = shoulder_angle
            data.qpos[elbow_id] = elbow_angle

        mujoco.mj_step(model, data)
        viewer.sync()

        cv2.imshow("Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
