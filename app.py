from flask import Flask, render_template, Response
import cv2
import pyttsx3
import numpy as np
from proto_sim import get_guidance  # Importing from best_path.py

app = Flask(__name__)

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Load YOLO model
net = cv2.dnn.readNet("yolov4-tiny.weights", "config.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)  # Webcam feed

def detect_objects():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        detected_objects = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (obj[:4] * [width, height, width, height]).astype("int")
                    x, y = int(center_x - w / 2), int(center_y - h / 2)

                    detected_objects.append((x, y, w, h))

                    # Warning for close objects
                    if w * h > (width * height) * 0.3:  
                        engine.say("Warning! Object very close.")
                        engine.runAndWait()

        # Get the best direction based on detected objects
        direction = get_best_direction(detected_objects, width, height)

        # Provide audio guidance
        if direction:
            engine.say(f"Move {direction}")
            engine.runAndWait()

        # Encode frame for web streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
