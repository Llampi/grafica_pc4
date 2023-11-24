import cv2
import mediapipe as mp
import numpy as np
import random
import math

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
contador_puntuacion = 0
# Para la cámara web
cap = cv2.VideoCapture("videog.mp4")
#cap = cv2.VideoCapture(0)


# no usado, creado para mover la silueta con el mouse
def drag_contour(event, x, y, flags, param):
    global contour

    if event == cv2.EVENT_MOUSEMOVE:
        # Actualiza el contorno
        contour = np.array([[x, y], [x+50, y], [x+50, y+50], [x, y+50]])

mensaje_inicial = "Presiona espacio para comenzar"

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:

 



  # Dibujar círculo adicional en el centro de la pantalla
  height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  circle_x = int(width / 2)
  circle_y = int(height / 2)
  circle_radius = 30
  circle_color = (0, 255, 0)  # Color verde en formato BGR
  imagen_inicio = np.ones((480, 640, 3), dtype=np.uint8) * 255
  cv2.putText(imagen_inicio, mensaje_inicial, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  cv2.imshow('MediaPipe Selfie Segmentation', imagen_inicio)
  while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # La tecla SPACE
                break
  while cap.isOpened():



    success, image = cap.read()
    if not success:
      print("Sin cámara.")
      break

    #inicio
    


    # bgr a rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # procesar imagen
    result = selfie_segmentation.process(image)

    # fondo blanco despues de segmentar
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    condition = result.segmentation_mask > 0.5
    condition = np.stack((condition,) * 3, axis=-1)
    bg_image = 255 * np.ones_like(image).astype(np.uint8)
    output_image = np.where(condition, image, bg_image)


    # contornos en la máscara de segmentación
    mask = np.uint8(result.segmentation_mask > 0.5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    



    # Dibujar contornos
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # Dibujar circulo
    cv2.circle(image, (circle_x, circle_y), circle_radius, circle_color, -1)
    #cv2.circle(image, (1000, 360), circle_radius, circle_color, -1)
    # Dibuja el círculo en la posición actual
    #cv2.circle(image, (circle_x, circle_y), radius=50, color=(0, 255, 0), thickness=-1)

    # Actualiza la posición Y del círculo para que se mueva hacia abajo
    
    user_mask = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.5
    
# Dibuja el círculo en la posición actual
    #cv2.circle(image, (circle_x, circle_y), radius=50, color=(0, 255, 0), thickness=-1)

    # Actualiza la posición Y del círculo para que se mueva hacia abajo
    circle_y += 20  # velocidad 

    if circle_y > height:
        circle_y = 0  # Reinicia la posición Y cuando el círculo sale de la pantalla
        circle_x = random.randint(50, int(width) - 50)  # Establece una nueva posición X aleatoria
        contador_puntuacion += 1
    y_coords, x_coords = np.ogrid[:result.segmentation_mask.shape[0], :result.segmentation_mask.shape[1]]
    # Crea una máscara para el círculo
    circle_mask = (x_coords - circle_x)**2 + (y_coords - circle_y)**2 <= circle_radius**2

    # Verifica si el círculo y la silueta se intersectan
    collision = np.any(np.logical_and(circle_mask, result.segmentation_mask > 0.5))

    if collision:
        
        print("El círculo está chocando con la silueta.")
        mensaje_colision = "Game Over: " + str(contador_puntuacion) + " puntos."
        #cv2.putText(image, mensaje_colision, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.imshow('MediaPipe Selfie Segmentation', image)
        imagen_final = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(imagen_final, mensaje_colision, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Juego terminado', imagen_final)
        #cv2.waitKey(5000)  # Pausa durante 3 segundos (3000 milisegundos)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # La tecla SPACE
                
                break
        #pass
        break
    else:
        #print("El círculo no está chocando con la silueta.")
        pass


    user_coords = np.argwhere(user_mask)
   
    user_y, user_x, _ = np.mean(user_coords, axis=0)
 

    # Calcular la distancia entre el usuario y el centro del círculo
    distance = math.sqrt((user_x - circle_x) ** 2 + (user_y - circle_y) ** 2)

    # Verificar si hay colisión (si la distancia es menor o igual al radio del círculo)
    if distance <= circle_radius*1.5:
        #print("¡Estás chocando con el círculo!")
        pass







    ##Mensaje
    # Definir el mensaje que quieres mostrar
    mensaje = "Puntuacion: " + str(contador_puntuacion)  # El mensaje que deseas mostrar

    # Configuración del texto
    font = cv2.FONT_HERSHEY_SIMPLEX  # Tipo de fuente
    posicion = (50, 50)  # Posición del texto en la imagen
    escala = 1  # Escala del texto
    color = (255, 0, 0)  # Color del texto en formato BGR
    grosor = 2  # Grosor del texto

    # Agregar texto a la imagen
    cv2.putText(image, mensaje, posicion, font, escala, color, grosor)

    cv2.imshow('MediaPipe Selfie Segmentation', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
