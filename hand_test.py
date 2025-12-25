import cv2
import mediapipe as mp

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Robot movement thresholds
x_center = 0.5  # normalized center
y_center = 0.5
threshold = 0.1  # how far from center before movement triggers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    command = "Idle"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Use index fingertip (landmark 8) as control point
        x, y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y

        # Determine movement
        if y < y_center - threshold:
            command = "Move Forward"
        elif y > y_center + threshold:
            command = "Move Backward"
        elif x < x_center - threshold:
            command = "Move Left"
        elif x > x_center + threshold:
            command = "Move Right"

    print(f"Robot Command: {command}")
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
