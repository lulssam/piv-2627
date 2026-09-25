'''
Exercício 2.3 - Transformações Geométricas
OpenCV methods:

* getRotationMatrix2D()
* warpAffine()

'''

import cv2
import numpy as np

theta = np.radians(30)

array1 = np.float32([
    (np.cos(theta), -np.sin(theta), 0),
    (np.sin(theta), np.cos(theta), 0)
])

caminho = "./imgs/fingerprint.jpg"
original = cv2.imread(caminho)

v = (original.shape[1]//2, original.shape[0]//2)
print(v)

w = 10

rotacao = cv2.getRotationMatrix2D(center=v, angle=w, scale=1.0)
rodada = cv2.warpAffine(original, rotacao, (original.shape[1], original.shape[0]))
cv2.imshow("Rotation", rotacao)


cv2.waitKey(0)
cv2.destroyAllWindows()