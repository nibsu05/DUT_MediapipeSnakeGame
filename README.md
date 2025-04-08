# Mediapipe Snake Game
![image](https://github.com/user-attachments/assets/332a8009-b017-433e-9eed-be0de9cf853d)

**DUT - Dự án GK Trí Tuệ Nhân Tạo - 2025**
- **Đề tài:** Điều khiển Game Rắn Săn Mồi qua AI nhận diện hình ảnh Webcam
- **Được thực hiện bởi:**
  
  + Lê Bá Bảo Thái - MSV: 102230133
  + Phạm Công Trung - MSV: 102210081
# DEMO dự án: 
https://www.youtube.com/watch?v=pk_OkAW4SvE
# Hướng dẫn
- _**collect_hand_data.py:**_ Thu thập mẫu lại từ đầu (nếu muốn) với phím N để đổi nhãn mẫu (UP, DOWN, LEFT, RIGHT) muốn thu thập, BACKSPACE để thu thập mẫu và lưu vào file hand_gesture_data.csv, bấm ESC để thoát quá trình thu thập mẫu
- _**Train_model.py:**_ Sau khi thu thập mẫu thì hãy train model lại một lần nữa để có thể tạo thành mô hình dự đoán được lưu tại file hand_gesture_model.pkl
- _**check_hand_gesture.py:**_ Kiểm thử mô hình dự đoán trước khi tích hợp vào Game, bấm ESC để thoát quá trình kiểm thử
- _**SnakeGame.py:**_ Kết quả dự án (bạn có thể chạy mà không cần thực hiện lại các file trên)
