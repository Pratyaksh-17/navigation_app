import cv2
import numpy as np

# Load YOLOv4 Tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "config.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize webcam
cap = cv2.VideoCapture(0)

def get_guidance(detections, frame_width):
    """
    Analyzes detected objects and determines the best movement direction.
    """
    left_count, center_count, right_count = 0, 0, 0
    min_distance = float('inf')
    danger_zone = False

    for detection in detections:
        x, y, w, h = detection[2:6]  # Bounding box values
        center_x = x + w // 2

        # Categorize object positions
        if center_x < frame_width * 0.3:
            left_count += 1
        elif center_x > frame_width * 0.7:
            right_count += 1
        else:
            center_count += 1

        # Determine closest object distance
        distance = 1000 / (w + 1)  # Approximate depth estimation
        if distance < min_distance:
            min_distance = distance

        # Set danger flag if object is very close
        if distance < 1.0:
            danger_zone = True

    # Determine best movement direction
    if danger_zone:
        guidance = "⚠️ WARNING! Obstacle too close! STOP!"
    elif center_count == 0:
        guidance = "✅ Move forward safely!"
    elif left_count > right_count:
        guidance = "➡️ Move RIGHT to avoid obstacles."
    elif right_count > left_count:
        guidance = "⬅️ Move LEFT to avoid obstacles."
    else:
        guidance = "⏸️ Adjust position carefully."

    return guidance, min_distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detections = []
    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO detections
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

    # Apply non-maxima suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    detected_objects = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw bounding boxes
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detected_objects.append([class_ids[i], confidence, x, y, w, h])

    # Get movement guidance
    guidance_text, min_distance = get_guidance(detected_objects, width)

    # Display guidance text
    cv2.putText(frame, guidance_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if "WARNING" in guidance_text else (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Navigation System", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
