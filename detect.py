import tensorflow.lite as tflite
import cv2
import numpy as np

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="tflite_model/coco_ssd_mobilenet_v1_1.0_quant.tflite")
interpreter.allocate_tensors()

# Load Labels
with open("tflite_model/labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match model input size
    input_details = interpreter.get_input_details()
    height, width = input_details[0]['shape'][1:3]
    frame_resized = cv2.resize(frame, (width, height))
    frame_resized = np.expand_dims(frame_resized, axis=0)

    # Run the model
    interpreter.set_tensor(input_details[0]['index'], frame_resized)
    interpreter.invoke()

    # Get the output
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Draw bounding boxes
    for detection in output_data[0]:
        class_id = int(detection[1])
        score = detection[2]
        if score > 0.5:
            label = labels[class_id]
            print(f"Detected: {label} with confidence {score:.2f}")

    # Show the camera feed
    cv2.imshow("Blind Navigation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
