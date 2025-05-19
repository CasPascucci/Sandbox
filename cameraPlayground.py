from ultralytics import YOLO
import cv2

# Load YOLOv8 model (use 'yolov8n.pt' for the smallest and fastest version)
model = YOLO('yolo11n.pt')  # Make sure the weights file downloads automatically or already exists

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, save=False, verbose=False)

    # Draw the detection results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
