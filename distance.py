import cv2
import numpy as np

# Load YOLOv4-Tiny model
yoloNet = cv2.dnn.readNet("yolov4-tiny.weights", "config.cfg")
model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load Class Names
with open(r"C:\Users\pratg\Robotics Project\Codes\coco.names", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    classes, scores, boxes = model.detect(frame, 0.4, 0.3)

    for (classid, score, box) in zip(classes.flatten(), scores.flatten(), boxes):
        label = f"{class_names[classid]}: {score:.2f}"
        cv2.rectangle(frame, box, (0, 255, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("YOLOv4-Tiny Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
