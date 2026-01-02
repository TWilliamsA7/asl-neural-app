// Define UI elements
const socket = io();
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const labelDiv = document.getElementById("prediction-label");
const confBar = document.getElementById("confidence-bar");
const confText = document.getElementById("confidence-text");

// Capture canvas for frame extraction
const captureCanvas = document.createElement("canvas");
captureCanvas.width = 640;
captureCanvas.height = 480;
const captureContext = captureCanvas.getContext("2d");

// Access Webcam
navigator.mediaDevices
  .getUserMedia({
    video: { width: 640, height: 480 },
  })
  .then((stream) => {
    video.srcObject = stream;
  });

// Loop: Capture frame and send via WebSocket
function sendFrame() {
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    // Draw current video frame to hidden capture canvas
    captureContext.drawImage(video, 0, 0, 640, 480);
    const data = captureCanvas.toDataURL("image/jpeg", 0.6);
    socket.emit("image_frame", data);
  } else {
    // Wait for video to be ready
    requestAnimationFrame(sendFrame);
  }
}

socket.on("prediction_response", (data) => {
  const pred = data.prediction;

  // Update UI
  labelDiv.innerText = pred.label;
  const precentage = (pred.confidence * 100).toFixed(1);
  confBar.style.width = precentage + "%";
  confText.innerText = `Confidence: ${precentage}%`;

  // Draw annotated image onto the canvas
  const img = new Image();
  img.onload = () => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(img, 0, 0, canvas.width, canvas.height);
    // Trigger next frame after current one is processed and displayed
    requestAnimationFrame(sendFrame);
  };
  img.src = data.annotated_image;
});

video.onloadeddata = () => {
  sendFrame();
};
