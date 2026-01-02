# ASL Neural App

A lightweight full-stack application that translates American Sign Language in real time using a convolutional neural network.

## Overview

This project utilizes a full structured machine learning pipeline via **Google Colab** to create an ASL recognizing Convolutional Neural Network. This model was then integrated into a lightweight **flask** application for live inference. This repository is a rewrite and restructing of a previous codebase for improved performance, maintainability, and scalability.

The core focus of this repository is:

- **Modular Code:** Implementing modules within the `src/` directory for clear separation of concerns (e.g., data handling, modeling, and evaluation).
- **Reproducibility:** Using Git and a strict `requirements.txt` file to ensure consistent and reproducible environments across all execution platforms.
- **Scalable Execution:** Facilitating a seamless transition to cloud resources (like Colab's GPUs/TPUs) for heavy computation and training.

## Features :star:

- Full Machine Learning Pipeline Notebook
  - Google Colab Environment Initialization
  - Data: Acquisition & Preprocessing
  - Data: Loading and Splitting
  - Model: Architecture
    - Weight Reloading from Previous Runs
  - Model: Training
    - Backup Restoration for Stability
  - Model: Evaluation
- Live Inference Loop
  - Image Preprocessing
  - Hand Keypoint Extraction
  - Real-time Predictions
- Flask Inference Application
  - Real-time Live Inference and Display
  - Websocket Communication
  - Simple, Clear UI

## How The App Works

1. The model is loaded from the specified path.
2. The flask application displays a video feed and sends frames via websockets.
3. This frame is recieved and preprocessed to create a prediction object that is returned back
4. This prediction object is used to update the frame and display the inference provided by the model.

## Usage

To run the pipeline notebook:

- Import `notebooks/ASLNN_Full_Pipeline.ipynb` to Google Colab and run the cells in order.
- There are specific places where credentials and secrets are required that should be provided by the user

```bash
# Runs Flask Application
python app/main.py
# Runs Simple Live Inference
python src/main.py
```

## Preview

![Application Screenshot: E](assets/example1.jpg "Capturing E Sign")
![Application Screenshot: U](assets/example2.jpg "Capturing U Sign")

## Contributing

Issues and pull requests are welcome.
