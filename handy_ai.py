import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# ===== SETTINGS =====
SCREEN_W, SCREEN_H = pyautogui.size()
SMOOTHING = 7
CLICK_THRESHOLD = 35

# ===== INITIALIZE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

pyautogui.FAILSAFE = False

def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # ===== INDEX FINGER TIP =====
        index_tip = lm[8]
        thumb_tip = lm[4]
        middle_tip = lm[12]

        x = np.interp(index_tip.x, [0.1, 0.9], [0, SCREEN_W])
        y = np.interp(index_tip.y, [0.1, 0.9], [0, SCREEN_H])

        # Smooth movement
        curr_x = prev_x + (x - prev_x) / SMOOTHING
        curr_y = prev_y + (y - prev_y) / SMOOTHING

        pyautogui.moveTo(curr_x, curr_y)

        prev_x, prev_y = curr_x, curr_y

        # ===== LEFT CLICK =====
        thumb_index_dist = distance(thumb_tip, index_tip) * w

        if thumb_index_dist < CLICK_THRESHOLD:
            pyautogui.click()
            cv2.putText(
                frame,
                "CLICK",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                4
            )
            cv2.waitKey(200)

        # ===== RIGHT CLICK =====
        thumb_middle_dist = distance(thumb_tip, middle_tip) * w

        if thumb_middle_dist < CLICK_THRESHOLD:
            pyautogui.rightClick()
            cv2.putText(
                frame,
                "RIGHT CLICK",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                4
            )
            cv2.waitKey(200)

    cv2.imshow("Handy AI - Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()