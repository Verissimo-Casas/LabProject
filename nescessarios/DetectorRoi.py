# Import necessary libraries
import torch
import cv2
import numpy as np

# Define function to track mouse movement
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create a window and set mouse callback function
cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='/Users/verissimocasas/yolov5morethanroi/yolov5s.pt', source='local')

# Initialize variables
count = 0
cap = cv2.VideoCapture(0)

# Start video capture and object detection
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    # Resize frame to 640x640
    frame = cv2.resize(frame, (640, 640))

    # Detect objects in frame using YOLOv5 model
    results = model(frame)

    # Iterate through detected objects and draw bounding boxes, labels, and center points
    for index, row in results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
        label = row['name']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = cv2.rectangle(frame, (390, 0), (640, 640), (0, 255, 0), 2)
        if 390 <= cx <= 640 and 0 <= cy <= 640:
            print(f"{label} is in the area")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Display frame in window
    cv2.imshow("ROI", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy window
cap.release()
cv2.destroyAllWindows()