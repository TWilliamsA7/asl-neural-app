// Define UI elements
const socket = io();
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const labelDiv = document.getElementById("prediction-label");
const confBar = document.getElementById("confidence-bar");
const confText = document.getElementById("confidence-text");

// Access Webcam
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  video.srcObject = stream;
});

// Loop: Capture frame and send via WebSocket
function sendFrame() {
  context.drawImage(video, 0, 0, 640, 480);
  const data = canvas.toDataURL("image/jpeg", 0.5);
  socket.emit("image_frame", data);
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
  img.onload = () => context.drawImage(img, 0, 0);
  img.src = data.annotated_image;

  setTimeout(sendFrame, 50);
});

setTimeout(sendFrame, 1000);
