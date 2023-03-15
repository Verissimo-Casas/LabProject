import cv2
import torch

model = torch.hub.load('yolov5', 'custom', path="/Users/verissimocasas/yolov5morethanroi/yolov5s.pt", source='local')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 640x640
    frame = cv2.resize(frame, (640, 640))

    # Detect objects in frame using object detector
    results = model(frame)
    objects = results.pandas().xyxy[0]

    # Iterate through detected objects and draw bounding boxes, labels, and center points
    for index, row in objects.iterrows():
        x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
        label = row['name']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Draw area of interest
    cv2.rectangle(frame, (390, 0), (640, 640), (255, 0, 0), 2)

    # Show frame
    cv2.imshow('ROI', frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()