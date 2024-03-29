import cv2
import time

cam = 2             # Camera
width = 640         # Largura
height = 480        # Altura
fps = 60            # FPS 25/30/50/60

codec = 0x47504A4D  # MJPG
brightness = 55     # Brilho
contrast = 130      # Contraste
saturation = 112    # Saturação

# Câmera Hikvision não tem essas funçoes
focus = 0           # Foco
sharpness = 0       # Nitidez
exposure = 0        # Exposição


# FPS Teste
start_time = time.time()
display_time = 10
fc = 0
p_fps = 0

# Conexão da câmera
camera = cv2.VideoCapture(cam)

# Configuração da câmera
camera.set(cv2.CAP_PROP_FOURCC, codec)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera.set(cv2.CAP_PROP_FPS, fps)
camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
camera.set(cv2.CAP_PROP_CONTRAST, contrast)
camera.set(cv2.CAP_PROP_SATURATION, saturation)

#camera.set(cv2.CAP_PROP_FOCUS, focus)
#camera.set(cv2.CAP_PROP_SHARPNESS, sharpness)
# camera.set(cv2.CAP_PROP_EXPOSURE, exposure)


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

    # Add contador de frames
    image = cv2.putText(frame, fps_disp, (10, 25),
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rotate = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    # Mostar imagem
    cv2.imshow('Ball_Cam', rotate)
    
    # Verifica a 'q' para fehcar a imagem
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Libera a imagem da câmera
camera.release()
# Fechamendo do imagem
cv2.destroyAllWindows()
