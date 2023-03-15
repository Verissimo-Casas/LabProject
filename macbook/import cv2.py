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


