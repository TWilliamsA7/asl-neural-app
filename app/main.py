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
from src.model_utils import load_model_once, load_normalization_parameters

# Define App and Socket
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

MODEL = None

def init_app():
    global MODEL
    load_normalization_parameters('../data/keypoint_norm_params.npz')
    MODEL = load_model_once('../models/best_fusion_model.keras')
