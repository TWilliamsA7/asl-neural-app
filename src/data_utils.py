import numpy as np
import cv2
import mediapipe as mp
import os
import tensorflow as tf

# --- Global Definitions (No expensive initialization here) ---
mp_hands = mp.solutions.hands
hands_detector = None # Will hold the initialized mp_hands.Hands() object

# --- Configuration (These are safe) ---
IMG_WIDTH, IMG_HEIGHT = 224, 224


def initialize_hands_detector():
    """Initializes the MediaPipe Hands object only once."""
    global hands_detector
    if hands_detector is None:
        print("Initializing MediaPipe Hands Detector...")
        # This is the line that caused the error, now safely inside a function
        hands_detector = mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.5
        )
    return hands_detector


def extract_keypoints(image_path):
    """
    Loads an image, extracts normalized 2D hand landmark keypoints (42 values),
    and returns the resized image for CNN input.
    """

    detector = initialize_hands_detector()

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        return None, None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image for consistent CNN input (MobileNetV2 standard)
    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Process the image with MediaPipe
    results = detector.process(image_rgb)
    
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