import pygame
import sys
import random
import time
import cv2
import numpy as np
import mediapipe as mp
import joblib  # Dùng để load model đã huấn luyện

# Load model đã huấn luyện
model = joblib.load('hand_gesture_model.pkl')  # Đảm bảo file này tồn tại
# Khởi tạo MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
# Khởi tạo Pygame và font
pygame.init()
pygame.font.init()

# Kích thước cửa sổ tổng thể
GAME_AREA_WIDTH = 800  # Khu vực chơi game bên trái
UI_PANEL_WIDTH = 400  # Khu vực UI bên phải
WINDOW_WIDTH = GAME_AREA_WIDTH + UI_PANEL_WIDTH  # 1200
WINDOW_HEIGHT = 700

GAME_WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("DUT - Snake Game")

# FPS mặc định (tốc độ rắn) sẽ được điều chỉnh ở Settings
SNAKE_SPEED = 20
fps_controller = pygame.time.Clock()

# Định nghĩa màu sắc
BLACK = pygame.Color(0, 0, 0)
GRAY = pygame.Color(100, 100, 100)
DARK_UI = pygame.Color(50, 50, 50)  # Màu nền cho UI panel
WHITE = pygame.Color(255, 255, 255)
GREEN = pygame.Color(0, 255, 0)
GREEN2 = pygame.Color(0, 152, 0)
RED = pygame.Color(255, 0, 0)
ORANGE = pygame.Color(255, 165, 0)  # Màu cam cho tường
YELLOW = pygame.Color(255, 255, 0)

# Thiết lập webcam
cap = cv2.VideoCapture(0)
CAM_WIDTH, CAM_HEIGHT = 360, 260

# Thêm vào phần màu sắc
ACTIVE_ARROW = pygame.Color(255, 255, 0)  # Màu khi phím được nhấn
INACTIVE_ARROW = pygame.Color(150, 150, 150)  # Màu mặc định

# Kích thước mũi tên
ARROW_SIZE = 30


def draw_arrow(surface, direction, color, center):
    x, y = center
    if direction == 'UP':
        points = [(x - ARROW_SIZE, y + ARROW_SIZE),
                  (x, y - ARROW_SIZE),
                  (x + ARROW_SIZE, y + ARROW_SIZE)]
    elif direction == 'DOWN':
        points = [(x - ARROW_SIZE, y - ARROW_SIZE),
                  (x, y + ARROW_SIZE),
                  (x + ARROW_SIZE, y - ARROW_SIZE)]
    elif direction == 'LEFT':
        points = [(x + ARROW_SIZE, y - ARROW_SIZE),
                  (x - ARROW_SIZE, y),
                  (x + ARROW_SIZE, y + ARROW_SIZE)]
    elif direction == 'RIGHT':
        points = [(x - ARROW_SIZE, y - ARROW_SIZE),
                  (x + ARROW_SIZE, y),
                  (x - ARROW_SIZE, y + ARROW_SIZE)]
    pygame.draw.polygon(surface, color, points)


# Hàm vẽ text
def draw_text(surface, text, size, color, center):
    font = pygame.font.SysFont('consolas', size)
    text_surface = font.render(text, True, color)
    rect = text_surface.get_rect(center=center)
    surface.blit(text_surface, rect)


# Hàm tạo nút bấm
def button(surface, msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(surface, ac, (x, y, w, h))
        if click[0] == 1 and action is not None:
            action()
            time.sleep(0.3)
    else:
        pygame.draw.rect(surface, ic, (x, y, w, h))
    draw_text(surface, msg, 20, BLACK, (x + w // 2, y + h // 2))


# ---------------- SETTINGS MENU ----------------
def settings_menu():
    global SNAKE_SPEED
    speeds = [10, 15, 20, 25, 30]
    speed_labels = ["Very Slow", "Slow", "Normal", "Fast", "Very Fast"]
    current_index = speeds.index(SNAKE_SPEED) if SNAKE_SPEED in speeds else 2

    # Thiết lập thông số của thanh kéo
    slider_width = 500
    slider_height = 10
    slider_x = WINDOW_WIDTH // 2 - slider_width // 2
    slider_y = 150
    steps = len(speeds)
    gap = slider_width / (steps - 1)
    dragging = False

    selecting = True
    while selecting:
        GAME_WINDOW.fill(GREEN2)
        # Tiêu đề
        draw_text(GAME_WINDOW, "SETTINGS", 40, YELLOW, (WINDOW_WIDTH // 2, 50))
        # Hiển thị tốc độ hiện tại
        draw_text(GAME_WINDOW, f"Current Speed: {speeds[current_index]} FPS", 24, WHITE, (WINDOW_WIDTH // 2, 100))

        # Vẽ thanh kéo (track)
        pygame.draw.line(GAME_WINDOW, WHITE,
                         (slider_x, slider_y + slider_height // 2),
                         (slider_x + slider_width, slider_y + slider_height // 2), 4)
        # Vẽ các nấc trên thanh kéo
        for i in range(steps):
            step_x = slider_x + i * gap
            pygame.draw.circle(GAME_WINDOW, GRAY, (int(step_x), slider_y + slider_height // 2), 8)

        # Vẽ "knob" theo nấc hiện hành
        knob_x = slider_x + current_index * gap
        pygame.draw.circle(GAME_WINDOW, YELLOW, (int(knob_x), slider_y + slider_height // 2), 12)

        # Hiển thị nhãn tốc độ dưới thanh kéo
        draw_text(GAME_WINDOW, f"{speed_labels[current_index]} ({speeds[current_index]} FPS)", 24, WHITE,
                  (WINDOW_WIDTH // 2, slider_y + 50))

        # Vẽ nút Back bên dưới
        btn_w, btn_h = 150, 60
        btn_x = WINDOW_WIDTH // 2 - btn_w // 2
        btn_y = slider_y + 100
        button(GAME_WINDOW, "Back", btn_x, btn_y, btn_w, btn_h, GRAY, WHITE, lambda: quit_menu("back"))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                selecting = False
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.USEREVENT:
                if event.custom == "back":
                    selecting = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Kiểm tra nếu nhấn chuột gần knob
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if ((mouse_x - knob_x) ** 2 + (mouse_y - (slider_y + slider_height // 2)) ** 2) ** 0.5 <= 15:
                    dragging = True
            if event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            if event.type == pygame.MOUSEMOTION and dragging:
                # Tính chỉ số nấc mới dựa theo vị trí chuột, đảm bảo không vượt quá phạm vi
                mouse_x = pygame.mouse.get_pos()[0]
                new_index = round((mouse_x - slider_x) / gap)
                new_index = max(0, min(steps - 1, new_index))
                if new_index != current_index:
                    current_index = new_index
                    SNAKE_SPEED = speeds[current_index]
        fps_controller.tick(15)


def set_speed(new_speed):
    global SNAKE_SPEED
    SNAKE_SPEED = new_speed


def quit_menu(action):
    # Sử dụng sự kiện USEREVENT để điều khiển việc thoát menu Settings
    pygame.event.post(pygame.event.Event(pygame.USEREVENT, custom=action))


# ---------------- TUTORIAL SCREEN ----------------
def tutorial_screen():
    running = True
    while running:
        GAME_WINDOW.fill(GREEN2)
        draw_text(GAME_WINDOW, "Tutorial", 40, YELLOW, (WINDOW_WIDTH // 2, 50))
        instructions = [
            "Dùng bàn tay biểu thị các con số để di chuyển.",
            " - Số 1: UP",
            " - Số 2: DOWN",
            " - Số 3: LEFT",
            " - Số 4: RIGHT",
            "Ăn thức ăn (màu trắng) để tăng điểm.",
            "Chế độ 'Không tường': rắn thoải mái di chuyển.",
            "Chế độ 'Có tường': bức tường màu cam bao quanh.",
            "Rắn sẽ thua khi tự chạm vào thân mình.",
            "UI bên phải hiển thị điểm, webcam và hướng mà webcam nhận diện được.",
            "Settings cho phép chỉnh tốc độ rắn."
        ]
        for i, line in enumerate(instructions):
            draw_text(GAME_WINDOW, line, 20, WHITE, (WINDOW_WIDTH // 2, 120 + i * 30))
        draw_text(GAME_WINDOW, "Nhấn phím bất kỳ để quay lại", 18, BLACK, (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                running = False
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
        fps_controller.tick(15)


# ---------------- GAME MODE MENU ----------------
def game_mode_menu():
    mode = None
    selecting = True
    while selecting:
        GAME_WINDOW.fill(GREEN2)
        # Tiêu đề: căn giữa ở trên màn hình
        draw_text(GAME_WINDOW, "Chọn chế độ chơi", 40, YELLOW, (WINDOW_WIDTH // 2, 100))

        # Cấu hình kích thước và vị trí nút bấm mới
        btn_width = 250
        btn_height = 60
        btn_x = WINDOW_WIDTH // 2 - btn_width // 2

        # Vị trí các nút theo chiều dọc (cách đều nhau)
        no_wall_btn_y = 200
        wall_btn_y = no_wall_btn_y + btn_height + 30
        back_btn_y = wall_btn_y + btn_height + 30

        button(GAME_WINDOW, "Không tường", btn_x, no_wall_btn_y, btn_width, btn_height, GRAY, WHITE,
               lambda: set_mode("no_wall"))
        button(GAME_WINDOW, "Có tường", btn_x, wall_btn_y, btn_width, btn_height, GRAY, WHITE, lambda: set_mode("wall"))
        # Nút quay lại để trở về menu chính
        button(GAME_WINDOW, "Back", btn_x, back_btn_y, btn_width, btn_height, GRAY, WHITE,
               lambda: pygame.event.post(pygame.event.Event(pygame.USEREVENT, custom='back')))

        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                selecting = False
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.USEREVENT:
                if event.custom in ["no_wall", "wall"]:
                    mode = event.custom
                    selecting = False
                elif event.custom == "back":
                    selecting = False
                    start_menu()
                    return
        fps_controller.tick(15)
    if mode is not None:
        score = game_loop(mode)
        game_over_screen(score, mode)


def set_mode(selected):
    pygame.event.post(pygame.event.Event(pygame.USEREVENT, custom=selected))

def game_loop(mode):
    wall_thickness = 10  # Độ dày tường (cho chế độ "wall")
    snake_pos = [100, 50]  # Tọa độ ban đầu của rắn
    snake_body = [list(snake_pos), [90, 50], [80, 50]]
    food_pos = [
        random.randrange(wall_thickness, (GAME_AREA_WIDTH - wall_thickness) // 10) * 10,
        random.randrange(wall_thickness, (WINDOW_HEIGHT - wall_thickness) // 10) * 10
    ]
    food_spawn = True
    direction = 'RIGHT'
    score = 0
    running = True

    # Khởi tạo các giá trị mặc định cho xác suất (nếu không có nhận diện thì hiển thị 0%)
    gesture_result = {"prediction": None,
                      "probabilities": {"up": 0, "down": 0, "left": 0, "right": 0}}

    while running:
        # Thêm xử lý sự kiện tại đây
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                cap.release()
                pygame.quit()
                sys.exit()
        # Đọc ảnh từ webcam
        ret, frame = cap.read()
        if not ret:
            continue

        # Nhận diện cử chỉ tay và vẽ bounding box lên frame
        gesture_result_new = detect_gesture(frame)
        if gesture_result_new is not None:
            gesture_result = gesture_result_new  # cập nhật nếu có kết quả nhận diện
            pred = gesture_result["prediction"]
            if pred == "up" and direction != "DOWN":
                direction = "UP"
            elif pred == "down" and direction != "UP":
                direction = "DOWN"
            elif pred == "left" and direction != "RIGHT":
                direction = "LEFT"
            elif pred == "right" and direction != "LEFT":
                direction = "RIGHT"

        # Di chuyển rắn theo hướng hiện tại
        if direction == 'UP':
            snake_pos[1] -= 10
        elif direction == 'DOWN':
            snake_pos[1] += 10
        elif direction == 'LEFT':
            snake_pos[0] -= 10
        elif direction == 'RIGHT':
            snake_pos[0] += 10

        # Xử lý chế độ "no_wall"
        if mode == "no_wall":
            if snake_pos[0] < wall_thickness:
                snake_pos[0] = GAME_AREA_WIDTH - 10 - wall_thickness
            elif snake_pos[0] > GAME_AREA_WIDTH - 10 - wall_thickness:
                snake_pos[0] = wall_thickness
            if snake_pos[1] < wall_thickness:
                snake_pos[1] = WINDOW_HEIGHT - 10 - wall_thickness
            elif snake_pos[1] > WINDOW_HEIGHT - 10 - wall_thickness:
                snake_pos[1] = wall_thickness

        # Xử lý chế độ "wall"
        if mode == "wall":
            if snake_pos[0] < wall_thickness or snake_pos[0] > GAME_AREA_WIDTH - 10 - wall_thickness:
                running = False
            if snake_pos[1] < wall_thickness or snake_pos[1] > WINDOW_HEIGHT - 10 - wall_thickness:
                running = False

        snake_body.insert(0, list(snake_pos))
        if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
            score += 1
            food_spawn = False
        else:
            snake_body.pop()

        if not food_spawn:
            food_pos = [
                random.randrange(wall_thickness, (GAME_AREA_WIDTH - wall_thickness) // 10) * 10,
                random.randrange(wall_thickness, (WINDOW_HEIGHT - wall_thickness) // 10) * 10
            ]
            food_spawn = True

        # Kiểm tra va chạm với thân rắn
        for block in snake_body[1:]:
            if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                running = False

        # Vẽ giao diện game: khu vực chơi và panel UI bên phải
        GAME_WINDOW.fill(GRAY)
        pygame.draw.rect(GAME_WINDOW, BLACK, pygame.Rect(0, 0, GAME_AREA_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(GAME_WINDOW, DARK_UI, pygame.Rect(GAME_AREA_WIDTH, 0, UI_PANEL_WIDTH, WINDOW_HEIGHT))

        # Vẽ tường (cho chế độ "wall")
        if mode == "wall":
            pygame.draw.rect(GAME_WINDOW, ORANGE, pygame.Rect(0, 0, GAME_AREA_WIDTH, wall_thickness))
            pygame.draw.rect(GAME_WINDOW, ORANGE, pygame.Rect(0, WINDOW_HEIGHT - wall_thickness, GAME_AREA_WIDTH, wall_thickness))
            pygame.draw.rect(GAME_WINDOW, ORANGE, pygame.Rect(0, 0, wall_thickness, WINDOW_HEIGHT))
            pygame.draw.rect(GAME_WINDOW, ORANGE, pygame.Rect(GAME_AREA_WIDTH - wall_thickness, 0, wall_thickness, WINDOW_HEIGHT))

        # Vẽ rắn và thức ăn
        for pos in snake_body:
            pygame.draw.rect(GAME_WINDOW, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(GAME_WINDOW, WHITE, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

        # Hiển thị điểm (Score)
        draw_text(GAME_WINDOW, f"Score: {score}", 24, WHITE, (GAME_AREA_WIDTH + UI_PANEL_WIDTH // 2, 50))

        # ---------------- Vẽ mũi tên điều khiển ----------------
        arrow_center_x = GAME_AREA_WIDTH + UI_PANEL_WIDTH // 2 - 75
        arrow_center_y = 235
        up_pressed = (gesture_result["prediction"] == "up")
        down_pressed = (gesture_result["prediction"] == "down")
        left_pressed = (gesture_result["prediction"] == "left")
        right_pressed = (gesture_result["prediction"] == "right")

        draw_arrow(GAME_WINDOW, 'UP', ACTIVE_ARROW if up_pressed else INACTIVE_ARROW,
                   (arrow_center_x, arrow_center_y - ARROW_SIZE * 2))
        draw_arrow(GAME_WINDOW, 'DOWN', ACTIVE_ARROW if down_pressed else INACTIVE_ARROW,
                   (arrow_center_x, arrow_center_y + ARROW_SIZE * 2))
        draw_arrow(GAME_WINDOW, 'LEFT', ACTIVE_ARROW if left_pressed else INACTIVE_ARROW,
                   (arrow_center_x - ARROW_SIZE * 2, arrow_center_y))
        draw_arrow(GAME_WINDOW, 'RIGHT', ACTIVE_ARROW if right_pressed else INACTIVE_ARROW,
                   (arrow_center_x + ARROW_SIZE * 2, arrow_center_y))
        pygame.draw.circle(GAME_WINDOW, INACTIVE_ARROW, (arrow_center_x, arrow_center_y), ARROW_SIZE // 2)

        # ---------------- Hiển thị phần trăm xác suất của các hướng ----------------
        prob_text_y = arrow_center_y + ARROW_SIZE * 3
        probs = gesture_result["probabilities"]
        draw_text(GAME_WINDOW, f"Up: {probs.get('up', 0)}%", 20, WHITE,
                  (GAME_AREA_WIDTH + UI_PANEL_WIDTH // 2 + 100, prob_text_y - 130))
        draw_text(GAME_WINDOW, f"Down: {probs.get('down', 0)}%", 20, WHITE,
                  (GAME_AREA_WIDTH + UI_PANEL_WIDTH // 2 + 100, prob_text_y + 30 - 130))
        draw_text(GAME_WINDOW, f"Left: {probs.get('left', 0)}%", 20, WHITE,
                  (GAME_AREA_WIDTH + UI_PANEL_WIDTH // 2 + 100, prob_text_y + 60 - 130))
        draw_text(GAME_WINDOW, f"Right: {probs.get('right', 0)}%", 20, WHITE,
                  (GAME_AREA_WIDTH + UI_PANEL_WIDTH // 2 + 100, prob_text_y + 90 - 130))

        # ---------------- Hiển thị Webcam (UI Panel) ----------------
        # Frame đã được xử lý (với bounding box) sẽ được chuyển đổi và hiển thị
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
        webcam_x = GAME_AREA_WIDTH + (UI_PANEL_WIDTH - CAM_WIDTH) // 2
        webcam_y = WINDOW_HEIGHT - CAM_HEIGHT - 10
        GAME_WINDOW.blit(frame_surface, (webcam_x, webcam_y))

        pygame.display.update()
        fps_controller.tick(SNAKE_SPEED)
    return score

def detect_gesture(frame):
    # Chuyển ảnh từ BGR sang RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Vẽ xương bàn tay (hand skeleton) lên frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            h, w, _ = frame.shape
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:  # 21 điểm * 2 (x, y)
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]

                # Giả sử thứ tự các lớp là ["down", "left", "right", "up"],
                # kiểm tra lại thứ tự model.classes_ nếu cần
                classes = model.classes_
                probabilities = {}
                for cls, p in zip(classes, proba):
                    probabilities[cls.lower()] = round(p * 100, 1)  # chuyển thành phần trăm

                # Trả về dự đoán và dictionary xác suất
                return {
                    "prediction": prediction.lower(),
                    "probabilities": probabilities
                }
    return None


# ---------------- GAME OVER SCREEN ----------------
def game_over_screen(score, mode):
    over = True
    while over:
        GAME_WINDOW.fill(GREEN2)
        draw_text(GAME_WINDOW, "GAME OVER", 50, RED, (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3))
        draw_text(GAME_WINDOW, f"Score: {score}", 30, WHITE, (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3 + 50))
        draw_text(GAME_WINDOW, "R: Replay | Q: Quit | B: Back", 24, WHITE,
                  (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3 + 100))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                over = False
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    over = False
                    # Chơi lại mà không cần chọn lại chế độ
                    score = game_loop(mode)
                    game_over_screen(score, mode)
                elif event.key == pygame.K_q:
                    over = False
                    cap.release()
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_b:
                    over = False
                    start_menu()
        fps_controller.tick(10)


# ---------------- START MENU ----------------
def start_menu():
    menu = True

    # Load và xử lý logo
    try:
        logo = pygame.image.load("Logo.jpg")  # Đảm bảo đúng đường dẫn
        logo = pygame.transform.scale(logo, (150, 150))
    except Exception as e:
        print("Lỗi load logo:", e)
        logo = None

    while menu:
        # Vẽ nền cho 2 phần
        GAME_WINDOW.fill(GREEN2, (0, 0, GAME_AREA_WIDTH, WINDOW_HEIGHT))  # Phần trái
        GAME_WINDOW.fill(DARK_UI, (GAME_AREA_WIDTH, 0, UI_PANEL_WIDTH, WINDOW_HEIGHT))  # Phần phải

        # Phần trái - Tiêu đề và các nút
        draw_text(GAME_WINDOW, "SNAKE GAME", 60, YELLOW, (GAME_AREA_WIDTH // 2, WINDOW_HEIGHT // 4))

        btn_width = 200
        btn_height = 60
        btn_x = (GAME_AREA_WIDTH - btn_width) // 2

        # Vị trí các nút
        start_btn_y = WINDOW_HEIGHT // 4 + 80
        tutorial_btn_y = start_btn_y + btn_height + 20
        settings_btn_y = tutorial_btn_y + btn_height + 20
        exit_btn_y = settings_btn_y + btn_height + 20

        # Vẽ các nút
        button(GAME_WINDOW, "Start", btn_x, start_btn_y, btn_width, btn_height, GRAY, WHITE,
               lambda: pygame.event.post(pygame.event.Event(pygame.USEREVENT, custom='start')))
        button(GAME_WINDOW, "Tutorial", btn_x, tutorial_btn_y, btn_width, btn_height, GRAY, WHITE,
               lambda: pygame.event.post(pygame.event.Event(pygame.USEREVENT, custom='tutorial')))
        button(GAME_WINDOW, "Settings", btn_x, settings_btn_y, btn_width, btn_height, GRAY, WHITE,
               lambda: pygame.event.post(pygame.event.Event(pygame.USEREVENT, custom='settings')))
        button(GAME_WINDOW, "Exit", btn_x, exit_btn_y, btn_width, btn_height, GRAY, WHITE,
               lambda: pygame.event.post(pygame.event.Event(pygame.QUIT)))

        # Phần phải - Thông tin giới thiệu
        info_x = GAME_AREA_WIDTH + 20
        info_y = 50
        draw_text(GAME_WINDOW, "Giới thiệu dự án", 30, WHITE, (info_x + (UI_PANEL_WIDTH - 40) // 2, info_y))
        info_lines = [
            "  Đề tài: Điều khiển Game Rắn Săn Mồi  ",
            "     bằng nhận điện hình ảnh Webcam    ",
            "",
            "Thành viên thực hiện dự án",
            "- Lê Bá Bảo Thái - 102230133",
            "- Phạm Công Trung - 102210081",
            "",
            "Cảm ơn đã theo dõi dự án này!"
        ]

        # Vẽ từng dòng text
        for i, line in enumerate(info_lines):
            draw_text(GAME_WINDOW, line, 18, WHITE, (info_x + 182, info_y + 50 + i * 30))

        # Vẽ logo PHÍA DƯỚI CÙNG
        if logo:
            # Tính vị trí (cách đáy 20px)
            logo_x = GAME_AREA_WIDTH + (UI_PANEL_WIDTH - logo.get_width()) // 2
            logo_y = WINDOW_HEIGHT - logo.get_height() - 50  # Cách đáy 20px
            GAME_WINDOW.blit(logo, (logo_x, logo_y))

        # Cập nhật màn hình 1 LẦN DUY NHẤT
        pygame.display.update()

        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                menu = False
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.USEREVENT:
                if event.custom == 'start':
                    menu = False
                    game_mode_menu()
                elif event.custom == 'tutorial':
                    tutorial_screen()
                elif event.custom == 'settings':
                    settings_menu()

        fps_controller.tick(15)


def main():
    start_menu()


if __name__ == '__main__':
    main()