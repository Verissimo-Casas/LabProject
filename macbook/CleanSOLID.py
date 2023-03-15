import torch
import cv2


class BoundingBox:
    def __init__(self, x1, y1, x2, y2, name):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.name = name

    def get_center(self):
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))


class ObjectDetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

    def detect_objects(self, image):
        results = self.model(image)
        return [BoundingBox(*row[['xmin', 'ymin', 'xmax', 'ymax', 'name']].values) for index, row in results.pandas().xyxy[0].iterrows()]


class VideoStream:
    def __init__(self, source):
        self.stream = cv2.VideoCapture(source)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.stream.read()
        if not ret:
            raise StopIteration()
        return frame


class Window:
    def __init__(self, name):
        self.name = name
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.handle_mouse_event)

    def handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print([x, y])

    def show_frame(self, frame):
        cv2.imshow(self.name, frame)

    def is_key_pressed(self, key):
        return cv2.waitKey(1) & 0xFF == ord(key)


class ObjectDetectionApp:
    def __init__(self, video_stream, object_detector, window):
        self.video_stream = video_stream
        self.object_detector = object_detector
        self.window = window

    def run(self):
        for frame in self.video_stream:
            objects = self.object_detector.detect_objects(frame)

            for obj in objects:
                cx, cy = obj.get_center()
                if 390 < cx < 640 and 0 < cy < 640:
                    print(f"{obj.name} is in the area")

                cv2.rectangle(frame, (obj.x1, obj.y1), (obj.x2, obj.y2), (0, 255, 0), 2)
                cv2.putText(frame, obj.name, (obj.x1, obj.y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            self.window.show_frame(frame)

            if self.window.is_key_pressed('q'):
                break


if __name__ == '__main__':
    model_path = '/Users/verissimocasas/yolov5morethanroi/yolov5s.pt'

    with VideoStream(0) as video_stream, Window('ROI') as window:
        object_detector = ObjectDetector(model_path)
        app = ObjectDetectionApp(video_stream, object_detector, window)
        app.run()

    cv2.destroyAllWindows()
