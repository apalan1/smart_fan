import cv2
import mediapipe as mp
from ultralytics import YOLO
import time


#For finger tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

def count_fingers(hand_landmarks, hand_label):
    finger_tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (check if it's extended)
    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

    # Other four fingers (tip is above the pip joint)
    for tip_id in finger_tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


#Directional Input
import time

last_print_time = 0

def trackBox(x1, y1, x2, y2, frame):
    global last_print_time
    current_time = time.time()

    if current_time - last_print_time >= 1:  # at least 1 second passed
        xMid = (x1 + x2)/2
        yMid = (y1 + y2)/2

        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        if (xMid > frame_center_x):
            print("left")
        else:
            print("right")

        if (yMid > frame_center_y):
            print("up")
        else:
            print("down")

        last_print_time = current_time  # reset timer


import time

last_finger_count = 1
start_hold_time = None
speed = 0

#Capture Fan Speed --> Want the hand to be up for 2 seconds to change the fan speed
def fanSpeed(finger_count):
    global last_finger_count
    global start_hold_time
    current_time = time.time()

    if last_finger_count != 0:
        if finger_count == last_finger_count:
            if start_hold_time is not None and time.time() - start_hold_time >= 2:
                # Finger count held for 2 seconds
                cv2.putText(frame, f'Held {finger_count} for 2s!',
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            last_finger_count = finger_count
            start_hold_time = time.time()




    #if finger_count == last_finger_count for 2 seconds 
        #display fan speed: finger_count
    #else if it ever changes, then update last_finger_count

    #Note: fan speed must be 0 or above

#Load model for person tracking
model = YOLO('yolov8n.pt')  # This will download the model if not cached


#start camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    #HAND
    
    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_info.classification[0].label  # 'Left' or 'Right'
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers
            finger_count = count_fingers(hand_landmarks, hand_label)
            cv2.putText(frame, f'{hand_label} Hand: {finger_count} fingers',
                        (10, 30 if hand_label == "Right" else 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            fanSpeed(finger_count)
    
    # Run detection (only 'person' class)
    results = model(frame, classes=[0], conf=0.6, verbose=False)

    # Draw boxes on detected persons
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = f'Person {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            trackBox(x1,y1,x2,y2,frame)

    cv2.imshow('Concurrent Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
