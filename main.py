import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# Increased confidence for maximum stability
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.85, 
    min_tracking_confidence=0.85
)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("VisionWrite AI", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("VisionWrite AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

canvas = None
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
draw_color = (255, 0, 255)
SMOOTHING = 0.4 # Higher smoothing = less jitter

# Stability variables
current_mode = "IDLE"
mode_buffer = []
BUFFER_SIZE = 3 # Mode must be consistent for 3 frames

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    if canvas is None: canvas = np.zeros((h, w, 3), np.uint8)

    # UI: Toolbar
    cv2.rectangle(frame, (0,0), (w, 80), (25, 25, 25), -1)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    for i, col in enumerate(colors):
        cv2.rectangle(frame, (20 + i*130, 15), (130 + i*130, 65), col, -1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 1. ALWAYS DRAW SKELETON
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lm = hand_landmarks.landmark
            
            # Detect fingers (True if extended)
            # Index(8), Middle(12), Ring(16), Pinky(20)
            f_index = lm[8].y < lm[6].y
            f_middle = lm[12].y < lm[10].y
            f_ring = lm[16].y < lm[14].y
            f_pinky = lm[20].y < lm[18].y

            # Smoothing coordinates
            rx, ry = int(lm[8].x * w), int(lm[8].y * h)
            curr_x = int(SMOOTHING * rx + (1 - SMOOTHING) * curr_x)
            curr_y = int(SMOOTHING * ry + (1 - SMOOTHING) * curr_y)

            # --- STABLE GESTURE DETECTION ---
            detected_mode = "IDLE"
            
            # FIST (All down) -> CLEAR
            if not f_index and not f_middle and not f_ring and not f_pinky:
                detected_mode = "CLEAR"
            # INDEX + MIDDLE (Peace) -> HOVER / COLOR PICK
            elif f_index and f_middle:
                detected_mode = "HOVER"
            # ONLY INDEX UP -> DRAW
            elif f_index and not f_middle:
                detected_mode = "DRAW"
            # PINKY UP -> ERASER
            elif f_pinky and not f_index:
                detected_mode = "ERASER"

            # Fill buffer to check for consistency
            mode_buffer.append(detected_mode)
            if len(mode_buffer) > BUFFER_SIZE:
                mode_buffer.pop(0)

            # Only change mode if buffer is consistent
            if len(mode_buffer) == BUFFER_SIZE and all(m == mode_buffer[0] for m in mode_buffer):
                current_mode = mode_buffer[0]

            # --- EXECUTE STABLE MODE ---
            if current_mode == "CLEAR":
                canvas = np.zeros((h, w, 3), np.uint8)
                prev_x, prev_y = 0, 0
            
            elif current_mode == "HOVER":
                prev_x, prev_y = 0, 0 # Line break!
                cv2.circle(frame, (curr_x, curr_y), 8, (255, 255, 255), 2)
                if curr_y < 80: # Color picker
                    if 20 < curr_x < 130: draw_color = (255, 0, 0)
                    elif 150 < curr_x < 260: draw_color = (0, 255, 0)
                    elif 280 < curr_x < 390: draw_color = (0, 0, 255)
                    elif 410 < curr_x < 520: draw_color = (0, 255, 255)

            elif current_mode == "DRAW":
                if prev_x == 0 and prev_y == 0: prev_x, prev_y = curr_x, curr_y
                cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), draw_color, 10)
                prev_x, prev_y = curr_x, curr_y

            elif current_mode == "ERASER":
                cv2.circle(canvas, (curr_x, curr_y), 50, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0

    # Display Mode Status
    cv2.putText(frame, f"STATUS: {current_mode}", (w-250, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Blend
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)
    cv2.imshow("VisionWrite AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()