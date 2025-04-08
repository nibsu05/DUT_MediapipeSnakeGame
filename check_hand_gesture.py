import cv2
import mediapipe as mp
import numpy as np
import joblib  # Dùng để load model đã lưu

# Load model đã huấn luyện
model = joblib.load('hand_gesture_model.pkl')  # Đảm bảo file này tồn tại

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Bật camera
cap = cv2.VideoCapture(0)

print("Đang chạy... Nhấn ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Trích xuất tọa độ (x, y) của các điểm
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:  # 21 điểm * 2 (x và y)
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data).max()

                # Hiển thị dự đoán và xác suất
                cv2.putText(frame, f"Gesture: {prediction} ({prob:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Nếu dự đoán là cử chỉ "UP" hoặc "DOWN", hiển thị thêm
                if prediction == "UP":
                    cv2.putText(frame, "Cử chỉ: Lên", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif prediction == "DOWN":
                    cv2.putText(frame, "Cử chỉ: Xuống", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Gesture Recognition", frame)

    # Nhấn ESC để thoát
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
