import tensorflow as tf
import numpy as np
import cv2
import os
import mediapipe as mp

MODEL_PATH = 'models/best_fusion_model.keras'
IMAGE_SIZE = (224, 224)
KEYPOINT_FEATURES = 42
NUM_CLASSES = 29
CLASS_LABELS = ['A', 'B', 'C', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SPACE', 'DELETE', 'NOTHING']

# Global variables for normalization parameters
KEYPOINT_MU = None
KEYPOINT_SIGMA = None

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


def load_normalization_parameters(path):
    """
    Loads the mean (mu) and sigma for Z-score normalization of keypoints 
    from the NPZ file created during training.
    """
    global KEYPOINT_MU, KEYPOINT_SIGMA
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normalization parameters not found at: {path}. Have you run the export script and copied the file?")
    
    # === CRITICAL LOADING LOGIC HERE ===
    
    # 1. Load the NPZ file. It returns a dictionary-like object.
    data = np.load(path)
    
    # 2. Access the arrays using the keys they were saved with ('mu', 'sigma')
    KEYPOINT_MU = data['mu']
    KEYPOINT_SIGMA = data['sigma']
    
    # 3. Check for correctness
    if KEYPOINT_MU.shape != (KEYPOINT_FEATURES,) or KEYPOINT_SIGMA.shape != (KEYPOINT_FEATURES,):
         raise ValueError(f"Normalization parameters shape mismatch. Expected (42,), got mu:{KEYPOINT_MU.shape}, sigma:{KEYPOINT_SIGMA.shape}")

    print("Keypoint normalization parameters loaded successfully.")




def preprocess_cnn_input(frame):
    """
        Resizes and normalizes the input frame for the CNN branch (image_input).
        Assumes frame is already loaded as a NumPy array (BGR format from cv2).
    """
    # 1. Resize to the target input size (224x224)
    img_resized = cv2.resize(frame, IMAGE_SIZE)

    # 2. Normalize pixel values to [0, 1] (Assuming your original preprocessing did this)
    img_normalized = img_resized
    
    # 3. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_final = np.expand_dims(img_normalized, axis=0)
    
    return img_final


def extract_keypoints_from_frame(frame, draw_landmarks=True):
    """
    Processes an image frame using MediaPipe to extract 42 hand keypoint features.
    
    Returns: 
        - keypoint_array (np.array): (1, 42) array of normalized (x, y) coordinates, 
          or None if no hand is detected.
        - frame (np.array): The frame with optional landmarks drawn.
    """
    # 1. Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Process the image and find hands
    results = hands.process(frame_rgb)
    
    keypoints = []
    
    if results.multi_hand_landmarks:
        # We only process the first detected hand (max_num_hands=1)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw landmarks on the frame (for visualization)
        if draw_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        # Extract normalized (x, y) coordinates (21 landmarks * 2 coords = 42 features)
        for landmark in hand_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
        
        # Ensure keypoint array is exactly 42 features
        if len(keypoints) != KEYPOINT_FEATURES:
            print(f"Warning: Expected {KEYPOINT_FEATURES} features, but got {len(keypoints)}. Skipping frame.")
            return None, frame

        # 3. Format keypoints for the model
        keypoint_data = np.array(keypoints).astype('float32')
        keypoint_data_normalized = (keypoint_data - KEYPOINT_MU) / KEYPOINT_SIGMA
        # Add batch dimension: (42,) -> (1, 42)
        keypoint_final = np.expand_dims(keypoint_data_normalized, axis=0)
        
        return keypoint_final, frame
    
    return None, frame # No hand detected


def run_prediction(model, image_input, keypoint_input):
    """Runs the dual-input prediction on the preprocessed data."""
    dual_input_data = {
        "image_input": image_input,     
        "keypoint_input": keypoint_input
    }
    
    raw_output = model.predict(dual_input_data, verbose=0)[0]
    predicted_index = np.argmax(raw_output)
    confidence = raw_output[predicted_index]
    predicted_label = CLASS_LABELS[predicted_index]
    
    return predicted_label, confidence, raw_output


def load_model_once(model_path):
    """Utility to load the Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ERROR: Model file not found at {model_path}. Please ensure the file is saved correctly.")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Tip: Ensure TensorFlow and Keras are correctly installed.")
        return None
    

def live_inference_loop(model):
    """Initializes webcam and runs real-time inference."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Starting Live Inference (Press 'q' to quit) ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a more intuitive selfie view
        frame = cv2.flip(frame, 1)
        
        # Extract keypoints and get the annotated frame
        keypoint_input, annotated_frame = extract_keypoints_from_frame(frame, draw_landmarks=True)
        
        display_text = "No Hand Detected"
        
        if keypoint_input is not None:
            # Prepare image input (224x224 and normalized)
            image_input = preprocess_cnn_input(frame)
            
            # Run prediction
            label, confidence, _ = run_prediction(model, image_input, keypoint_input)
            
            display_text = f"Sign: {label} (Conf: {confidence:.2f})"
            
        # Display the results on the frame
        cv2.putText(annotated_frame, 
                    display_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2, 
                    cv2.LINE_AA)

        cv2.imshow('ASL Fusion Inference', annotated_frame)
        
        # Break the loop on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def static_inference(model, image_path):
    """Runs inference on a single static image file."""
    print(f"\n--- Running Static Inference on: {image_path} ---")
    
    # 1. Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load static image from {image_path}")
        return

    # 2. Extract Keypoints
    keypoint_input, _ = extract_keypoints_from_frame(frame, draw_landmarks=False) # No drawing needed for static file
    
    if keypoint_input is not None:
        # 3. Prepare CNN Input
        image_input = preprocess_cnn_input(frame)
        
        # 4. Run Prediction
        label, confidence, _ = run_prediction(model, image_input, keypoint_input)
        
        print("\n--- STATIC INFERENCE RESULT ---")
        print(f"Predicted ASL Sign: {label}")
        print(f"Confidence: {confidence:.4f}")
        print("------------------------------")
    else:
        print("Prediction failed: No hand detected in the static image.")