import numpy as np
import cv2
import mediapipe as mp
import os
import tensorflow as tf

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# Use static_image_mode=True since we are processing stored images (not a live video stream)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224

def extract_keypoints(image_path):
    """
    Loads an image, extracts normalized 2D hand landmark keypoints (42 values),
    and returns the resized image for CNN input.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        return None, None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image for consistent CNN input (MobileNetV2 standard)
    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Process the image with MediaPipe
    results = hands.process(image_rgb)
    
    keypoints = []
    
    if results.multi_hand_landmarks:
        # Assuming only one hand is detected (max_num_hands=1)
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                # Append normalized X and Y coordinates (21 landmarks * 2 coords = 42 features)
                keypoints.append(landmark.x)
                keypoints.append(landmark.y)
        
        if len(keypoints) == 42:
            return np.array(keypoints, dtype=np.float32), resized_image
    
    # Return None if no valid hand detection occurred
    return None, None