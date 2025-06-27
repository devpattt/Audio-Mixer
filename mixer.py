import cv2
import mediapipe as mp
import pygame
import math

# === Audio Setup ===
pygame.mixer.init()
pygame.mixer.music.load('music.mp3')  # Your audio file here
pygame.mixer.music.play(-1)
pygame.mixer.music.set_volume(0.5)

# === MediaPipe Hand Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Webcam Setup ===
cap = cv2.VideoCapture(0)

# Simulated pitch level
pitch_level = 1.0  # default pitch (1.0x)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    hand_positions = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Detect left/right
            label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb and index tip landmarks
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Draw line and dots
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)

            # Midpoint
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            hand_positions.append((mid_x, mid_y))

            # Distance
            distance = math.hypot(x2 - x1, y2 - y1)
            value = min(max((distance - 20) / 180, 0.0), 1.0)

            if label == "Right":
                pygame.mixer.music.set_volume(value)
                cv2.putText(frame, f'Volume: {int(value * 100)}%', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                pitch_level = 0.5 + value * 1.5  # Range: 0.5x to 2.0x
                cv2.putText(frame, f'Pitch (sim): {pitch_level:.2f}x', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)

    # === Fancy Spectrum Between Hands ===
    if len(hand_positions) == 2:
        x1, y1 = hand_positions[0]
        x2, y2 = hand_positions[1]
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

        num_bars = 30
        bar_width = 6
        dx = (x2 - x1) / num_bars
        dy = (y2 - y1) / num_bars

        for i in range(num_bars):
            bx = int(x1 + dx * i)
            by = int(y1 + dy * i)

            bar_height = int(50 + math.sin(i * 0.5 + pygame.time.get_ticks() * 0.005) * 40 * pitch_level)
            hue = int((i / num_bars) * 255)
            color = (int(hue / 2), 255 - hue, 100 + int(hue / 3))

            cv2.rectangle(frame, (bx - bar_width // 2, by - bar_height),
                          (bx + bar_width // 2, by + bar_height), color, -1)

    # Show video
    cv2.imshow("Dual Hand Audio Control with Spectrum", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
