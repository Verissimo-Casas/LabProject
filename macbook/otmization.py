def object_detector(model_path):
    return torch.hub.load('yolov5', 'custom', path=model_path, source='local')

def detect_objects(model, frame):
    results = model(frame)
    return results.pandas().xyxy[0]

def is_object_in_area(cx, cy):
    return 390 <= cx <= 640 and 0 <= cy <= 640

def draw_bounding_box(image, bbox, label):
    x1, y1, x2, y2 = map(int, bbox[['xmin', 'ymin', 'xmax', 'ymax']])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    return (cx, cy)

def draw_area_of_interest(image):
    cv2.rectangle(image, (390, 0), (640, 640), (255, 0, 0), 2)

def show_frame(image):
    cv2.imshow('ROI', image)

def destroy_window():
    cv2.destroyAllWindows()
