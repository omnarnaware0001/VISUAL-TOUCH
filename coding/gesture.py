import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os

# --- DOWNLOAD MODEL IF MISSING ---
# The new MediaPipe requires a physical .task file to run.
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    import urllib.request
    print("Downloading required AI model (hand_landmarker.task)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# --- CONFIGURATION ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize Camera
cap = cv2.VideoCapture(0)
w_screen, h_screen = pyautogui.size()
w_cam, h_cam = 640, 480
cap.set(3, w_cam)
cap.set(4, h_cam)

# Smoothing Variables
ploc_x, ploc_y = 0, 0
cloc_x, cloc_y = 0, 0
smoothening = 5

print("System Active. Press 'Esc' to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect hands (Timestamp is required in VIDEO mode)
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        detection_result = landmarker.detect_for_video(mp_image, timestamp)
        
        # Access the landmarks
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                
                # Draw landmarks manually (since drawing_utils might be missing)
                for landmark in hand_landmarks:
                    x = int(landmark.x * w_cam)
                    y = int(landmark.y * h_cam)
                    cv2.circle(image, (x, y), 5, (255, 0, 255), cv2.FILLED)

                # Get Index Finger Tip (Index 8) and Thumb Tip (Index 4)
                index_tip = hand_landmarks[8]
                thumb_tip = hand_landmarks[4]
                
                x1, y1 = int(index_tip.x * w_cam), int(index_tip.y * h_cam)
                x2, y2 = int(thumb_tip.x * w_cam), int(thumb_tip.y * h_cam)

                # --- MOVE MOUSE ---
                # Map coordinates
                x3 = np.interp(x1, (0, w_cam), (0, w_screen))
                y3 = np.interp(y1, (0, h_cam), (0, h_screen))

                # Smooth
                cloc_x = ploc_x + (x3 - ploc_x) / smoothening
                cloc_y = ploc_y + (y3 - ploc_y) / smoothening

                try:
                    pyautogui.moveTo(cloc_x, cloc_y)
                except:
                    pass
                    
                ploc_x, ploc_y = cloc_x, cloc_y

                # --- CLICK ---
                # Calculate distance (Hypotenuse)
                distance = np.hypot(x2 - x1, y2 - y1)
                if distance < 30:
                    cv2.circle(image, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()

        cv2.imshow('Gesture Mouse (New API)', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()