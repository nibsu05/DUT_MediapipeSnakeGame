# collect_hand_data.py
import cv2
import mediapipe as mp
import numpy as np
import csv
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Nhãn: up, down, left, right
labels = ['up', 'down', 'left', 'right']
data = []

cap = cv2.VideoCapture(0)
label_index = 0
sample_count = 0

print("Nhấn SPACE để thu thập 1 khung hình, nhấn ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:
                cv2.putText(frame, f"Label: {labels[label_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2)

    cv2.imshow("Collecting Data", frame)
    key = cv2.waitKey(1)

    if key == ord(' '):  # SPACE
        if result.multi_hand_landmarks:
            print(f"Collected: {labels[label_index]}")
            data.append([labels[label_index]] + landmarks)
            sample_count += 1
    elif key == ord('n'):
        label_index = (label_index + 1) % len(labels)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Lưu vào file CSV
with open('hand_gesture_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"Đã thu thập tổng cộng {sample_count} mẫu.")
print("Dữ liệu đã được lưu vào hand_gesture_data.csv")
