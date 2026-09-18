# AULA 18 DE SETEMBRO
'''
Exercício 1.1 - Leitura e visualização de uma imagem;
OpenCV methods:

* imread()
* imshow()
* waitKey()
* destroyWindow()
* destroyAllWindows()
* cvtColor()
* namedWindow()
'''

import cv2
import numpy as np
caminho = "./imgs/florest.jpg" # caminho da imagem

img = cv2.imread(caminho) # ler imagem
cv2.imshow("Original", img) # Nome da Janela, imagem

cinzento = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converter para preto e branco
cv2.imshow("Cinzento", cinzento) # mostrar nova imagem

while cv2.waitKey(30) & 0xFF != ord("q"):
    pass

cv2.destroyAllWindows()

