
'''
Exercício 1.2 - Leitura e visualização de um vídeo ou imagens de uma câmara;
OpenCV methods

* VideoCapture()
* VideoCapture.get()
* VideoCapture.read()
* VideoCapture.release()
'''
import cv2

video = cv2.VideoCapture("./videos/vtest.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

print("largura: ", video.get(cv2.CAP_PROP_FRAME_WIDTH))
print("altura: ", video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame count: ", video.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame rate: ", fps)

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    cv2.imshow("video", frame)
    
    if cv2.waitKey(int(1000 / fps)) == ord("q"): # waitkey(1) impoe velocidade de 1 milisegundo então video parece mais rapido
        break
    
video.release()
cv2.destroyAllWindows()

# web cam
camera = cv2.VideoCapture(0)

fps = camera.get(cv2.CAP_PROP_FPS)

print("largura: ", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print("altura: ", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame count: ", camera.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame rate: ", fps)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    cv2.imshow("camera", frame)
    
    if cv2.waitKey(int(1000 / fps)) == ord("q"): # waitkey(1) impoe velocidade de 1 milisegundo então video parece mais rapido
        break
    
camera.release()
cv2.destroyAllWindows()