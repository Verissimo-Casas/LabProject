import torch
import cv2
import numpy as np


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)


model = torch.hub.load('yolov5', 'custom', path='/Users/verissimocasas/yolov5morethanroi/yolov5s.pt', source='local') 
count=0
cap=cv2.VideoCapture(0)


while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame=cv2.resize(frame,(640,640))
    results = model(frame)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        label=(row['name'])
#        print(d)
        cx=int((x1+x2)/2)
        cy=int((y1+y2)/2)
        area =cv2.rectangle(frame,(390,0),(640,640),(0,255,0),2)
        if cx>390 and cy>0 and cx<640 and cy<640:
            print(label + " is in the area")

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        
        cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
        cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
    # area = [[390, 0], [640, 390]]
    cv2.imshow("ROI",frame)
    if cv2.waitKey(1)&0xFF== ord('q'):
        break
cap.release()
#stream.release()
cv2.destroyAllWindows()
