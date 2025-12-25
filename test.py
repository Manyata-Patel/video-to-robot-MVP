import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
print("Setup successful!")
cap.release()
