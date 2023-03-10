# Importar las bibliotecas necesarias
import cv2


def areaInteres(image):
    ''' Definir un área de interés (ROI, por sus siglas en inglés) '''

    
    img = cv2.imread(image) # Cargar una imagen con OpenCV
    y = 200 # (x, y) es la coordenada de la esquina superior izquierda de la región
    x = 200 # (w, h) son las dimensiones de la región.
    roi = img[y:y+400, x:x+300] # Seleccionar una región en la imagen


    cv2.imshow("IMG", img) # Mostrar la imagen original
    cv2.imshow("ROI", roi) # Mostrar la imagen resultante
    cv2.waitKey(0) # espera a que se presione una tecla antes de cerrar la ventana que muestra la imagen
    cv2.destroyAllWindows() # cierra todas las ventanas de imagen abiertas por OpenCV

if __name__ == '__main__':
    areaInteres("yolov5/data/images/bus.jpg")

