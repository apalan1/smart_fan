import cv2
import mediapipe as mp
from ultralytics import YOLO
import time


# === For finger tracking ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

def count_fingers(hand_landmarks, hand_label):
    finger_tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (check if extended)
    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

    # Other fingers
    for tip_id in finger_tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# === Person Tracking Box Logic ===
last_print_time = 0
def trackBox(x1, y1, x2, y2, frame):
    global last_print_time
    current_time = time.time()

    if current_time - last_print_time >= 1:
        xMid = (x1 + x2) / 2
        yMid = (y1 + y2) / 2

        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        print("LEFT" if xMid > frame_center_x else "RIGHT",
              "|",
              "UP" if yMid > frame_center_y else "DOWN")

        last_print_time = current_time

# === YOLO model load ===
model = YOLO('yolov8n.pt')

# === Camera start ===
cap = cv2.VideoCapture(0)

saved_speed = 1
last_count = None
count_start_time = None
HOLD_TIME = 2  # seconds

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # === Hand tracking ===
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_info.classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks, hand_label)

            if finger_count != last_count:
                last_count = finger_count
                count_start_time = time.time()
            else:
                if time.time() - count_start_time >= HOLD_TIME and finger_count != saved_speed and finger_count > 0:
                    saved_speed = finger_count


            cv2.putText(frame, f'{hand_label} Hand: {finger_count} fingers',
                        (10, 30 if hand_label == "Right" else 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
  
    cv2.putText(frame, f'Saved Speed: {saved_speed}',
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 255, 0), 2)

    # === Person detection ===
    results = model(frame, classes=[0], conf=0.6, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = f'Person {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            trackBox(x1, y1, x2, y2, frame)

    # === Show window ===
    cv2.imshow('Concurrent Tracking', frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
