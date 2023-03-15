import cv2
import torch


class ObjectDetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

    def detect(self, image):
        results = self.model(image)
        return results.pandas().xyxy[0]

    def draw_bounding_box(self, image, bbox, label):
        x1, y1, x2, y2 = map(int, bbox[['xmin', 'ymin', 'xmax', 'ymax']])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        return (cx, cy)

class MouseTracker:
    def __init__(self):
        self.mouse_coords = None

    def track_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_coords = [x, y]

class Application:
    def __init__(self, model_path):
        self.detector = ObjectDetector(model_path)
        self.tracker = MouseTracker()

    def run(self):
        # Create a window and set mouse callback function
        cv2.namedWindow('ROI')
        cv2.setMouseCallback('ROI', self.tracker.track_mouse)

        # Initialize variables
        cap = cv2.VideoCapture(0)

        # Start video capture and object detection
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 640x640
            frame = cv2.resize(frame, (640, 640))

            # Detect objects in frame using YOLOv5 model
            detections = self.detector.detect(frame)

            # Iterate through detected objects and draw bounding boxes, labels, and center points
            for index, row in detections.iterrows():
                label = row['name']
                bbox = row[['xmin', 'ymin', 'xmax', 'ymax']]
                cx, cy = self.detector.draw_bounding_box(frame, bbox, label)
                if 390 <= cx <= 640 and 0 <= cy <= 640:
                    print(f"{label} is in the area")

            # Display frame in window
            cv2.imshow("ROI", frame)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and destroy window
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = '/Users/verissimocasas/yolov5morethanroi/yolov5s.pt'
    app = Application(model_path)
    app.run()
