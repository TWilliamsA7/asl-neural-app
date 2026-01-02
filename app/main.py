import os
import sys
import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Add src to path to reuse inference logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import dependencies from src
from src.model_utils import load_model_once, load_normalization_parameters, extract_keypoints_from_frame, preprocess_cnn_input, run_prediction

# Define App and Socket
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

MODEL = None

def init_app():
    global MODEL
    load_normalization_parameters('../data/keypoint_norm_params.npz')
    MODEL = load_model_once('../models/best_fusion_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image_frame')
def handle_image(data):
    # Decode base64 image
    header, encoded = data.split(',', 1)
    data = base64.b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)

    # Use inference logic
    keypoint_input, annotated_frame = extract_keypoints_from_frame(frame, draw_landmarks=True)

    result = {'label': 'No Hand', 'confidence': 0}

    if keypoint_input is not None:
        image_input = preprocess_cnn_input(frame)
        label, confidence, _ = run_prediction(MODEL, image_input, keypoint_input)
        result = {'label': label, 'confidence': float(confidence)}

    # Encode annotated frame to return to UI
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    emit('prediction_response', {
        'prediction': result,
        'annotated_image': f"data:image/jpeg;base64, {annotated_base64}"
    })

if __name__ == "__main__":
    init_app()
    socketio.run(app, debug=True, port=5000)
