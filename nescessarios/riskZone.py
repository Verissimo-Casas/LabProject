import cv2
import time
import numpy as np

# setup video capture

cam = 0
CODEC = 0x47504A4D  # MJPG

# FPS Teste
start_time = time.time()
display_time = 10
fc = 0
p_fps = 0


# Conexão da câmera
camera = cv2.VideoCapture(cam)

# Configuração da câmera
camera.set(cv2.CAP_PROP_FOURCC, CODEC)

frame = camera.read()

# Loop para mostrar imagem

while camera.isOpened():
    ret, frame = camera.read()
    
    if not ret:
        break
    
    fc+=1

    TIME = time.time() - start_time
    
    if (TIME) >= display_time :
        p_fps = fc / (TIME)
        fc = 0
        start_time = time.time()
    
    fps_disp = "FPS: "+str(p_fps)[:5]

    # show fps
    cv2.putText(frame, fps_disp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


    #draw a rectangle
    zone_color = (0,200,0)
    zone_thickness = 2
    frame = cv2.rectangle(frame, (0, 0), (310, 550), zone_color, zone_thickness)
    alpha = 0.5
    frame = cv2.addWeighted(frame, alpha, frame, 1 - alpha, 0)

    # show frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

camera.release()
cv2.destroyAllWindows()

def riskZone(image, bb_list):
    
    zone_color = (0, 255, 0)
    zone_thickness = 2
    zone = cv2.rectangle(image, (100, 100), (200, 200), zone_color, zone_thickness)

    for bb in bb_list:
        if bb[0] > 100 and bb[0] < 200 and bb[1] > 100 and bb[1] < 200:
            print("Pessoa na zona de risco")
            return True
    return False