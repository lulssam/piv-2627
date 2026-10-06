'''
Exercício 3.3 – Operadores morfológicos;
OpenCV methods: 
* getStructuringElement() 
* morphologyEx()
* dilate()
* erode().
'''

'''
Transformações morfológicas são operações simples baseadas
na estrutura de uma imagem, frequentemente utilizadas numa imagem binária.

'''

import cv2
import matplotlib.pyplot as plt

caminho = "./imgs/lily-lotus-flowers.jpg"
img = cv2.imread(caminho)

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eroded = cv2.erode(img_grey.copy(), None, iterations=3)

cv2.imshow("erode", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()