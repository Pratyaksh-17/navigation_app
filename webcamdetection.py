import cv2

cap = cv2.VideoCapture(0)  # Try 1 instead of 0 if it doesn't work

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Test Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close
        break

cap.release()
cv2.destroyAllWindows()
