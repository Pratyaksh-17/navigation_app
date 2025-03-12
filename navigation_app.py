from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from gtts import gTTS
import os
import threading

app = Flask(__name__)

# Global variables to control when video and audio should run
video_active = False
audio_active = False
cap = None  # Webcam will only be activated when needed

# Load YOLO Model
net = cv2.dnn.readNet("yolov4-tiny.weights", "config.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

def get_navigation_guidance(detections, frame_width):
    """
    Determines the best navigation path based on detected objects.
    """
    left_count, center_count, right_count = 0, 0, 0
    min_distance = float("inf")
    detected_objects = []

    for detection in detections:
        x, y, w, h = detection[2:6]  
        center_x = x + w // 2
        label = classes[detection[0]]

        detected_objects.append(label)

        if center_x < frame_width * 0.3:
            left_count += 1
        elif center_x > frame_width * 0.7:
            right_count += 1
        else:
            center_count += 1

        # Estimate distance
        distance = 1000 / (w + 1)
        if distance < min_distance:
            min_distance = distance

    # Decide movement based on object distribution
    if min_distance < 1.0:
        guidance = "⚠️ WARNING! Object very close! STOP!"
    elif center_count == 0:
        guidance = "✅ Move forward safely."
    elif left_count > right_count:
        guidance = "➡️ Move RIGHT to avoid obstacles."
    elif right_count > left_count:
        guidance = "⬅️ Move LEFT to avoid obstacles."
    else:
        guidance = "⏸️ Adjust position carefully."

    if audio_active:
        tts = gTTS(guidance, lang="en")
        tts.save("guidance.mp3")
        os.system("start guidance.mp3")  # Windows (Use "mpg321 guidance.mp3" for Linux)

    return guidance

def generate_video():
    """
    Captures video from the webcam and overlays object detection.
    """
    global cap, video_active

    if cap is None:
        cap = cv2.VideoCapture(0)  # Start the camera only when required

    while video_active:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Convert frame to blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        detections = []
        boxes, confidences, class_ids = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                    x, y = int(center_x - w / 2), int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    detections.append([class_id, confidence, x, y, w, h])

        # Apply Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected_objects = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                detected_objects.append([class_ids[i], confidences[i], x, y, w, h])

                # Draw bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(confidences[i] * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Get movement guidance
        guidance = get_navigation_guidance(detected_objects, width)

        # Display guidance text
        cv2.putText(frame, guidance, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if "WARNING" in guidance else (0, 255, 0), 2)

        # Convert to JPEG format
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/")
def index():
    """
    Renders the homepage.
    """
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """
    Starts video streaming only when requested.
    """
    global video_active
    video_active = True
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stop_video")
def stop_video():
    """
    Stops the video feed.
    """
    global video_active, cap
    video_active = False
    if cap:
        cap.release()
        cap = None
    return "Video stopped"

@app.route("/play_audio")
def play_audio():
    """
    Triggers AI-generated voice guidance.
    """
    global audio_active
    audio_active = True
    return "Audio guidance activated"

@app.route("/stop_audio")
def stop_audio():
    """
    Stops the AI-generated voice guidance.
    """
    global audio_active
    audio_active = False
    return "Audio guidance stopped"

@app.before_request
def auto_start():
    """
    Automatically starts Flask when the website is accessed.
    """
    global video_active
    if not video_active:
        threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5000, "debug": True}).start()
        video_active = True

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
