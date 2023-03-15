import cv2
import detector1


def capture(camera_index):

    # Crea un objeto de captura de cámara
    cap = cv2.VideoCapture(camera_index)

    # Loop hasta que se presione la tecla 'q'
    while True:
        # Lee un fotograma desde la cámara
        ret, frame = cap.read()
        
        
        # Si se ha capturado correctamente, muestra el fotograma
        if ret:
            bb_list = detector1.detect(frame)
            detector1.draw(frame, bb_list)
            detector1.area_pts(frame)
            
            detector1.checkCELL(frame, bb_list)
            frame = cv2.resize(frame, (int(1*frame.shape[1]),int(1*frame.shape[0])))
            
            
            #detector1.area(frame)
            # Muestra el fotograma en una ventana
            cv2.imshow('Camera', frame)
        
        # Si se presiona la tecla 'q', salir del loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera el objeto de captura y cierra la ventana
    cap.release()
    cv2.destroyAllWindows()

conectar = 'rtsp://admin:asdqwe123@192.168.10.101:554/cam/realmonitor?channel=1&subtype=1'

if __name__ == '__main__':
    #capture("video/Pexels Videos 1903289.mp4")
    capture(0)
    #capture('video/havana-cuba.mp4')
