import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

x_center = 0.5  
y_center = 0.5
threshold = 0.1 

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

        # index fingertip = landmark 8 and its used as control point
        x, y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y

        if y < y_center - threshold:
            command = "Move Forward"
        elif y > y_center + threshold:
            command = "Move Backward"
        elif x < x_center - threshold:
            command = "Move Left"
        elif x > x_center + threshold:
            command = "Move Right"

    print("Robot Command: " + command)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        breakq

cap.release()
cv2.destroyAllWindows() # MWAHAHAHAHA (jk lol)
