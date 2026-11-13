'''
Exercício 3.5 – Extracção de características;
OpenCV methods: contourArea(); moments(); arcLength();
boundingRect(); connectedComponentsWithStats()
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

caminho = "./imgs/formas.png"
imagem = cv2.imread(caminho)

# converter para escala de cinzentos
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# aplicar threshold
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

# encontrar contornos
contours,hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# desenhar contornos
imagem_contornos = imagem.copy()
cv2.drawContours(imagem_contornos, contours, -1, (0, 255, 0), 2)

'''
Moments são valores matemáticos que permitem descrever as características de uma forma numa imagem: 
* o tamanho 
* o centro
* algumas características da forma de um objeto numa imagem.
'''
if contours:
    cnt = max(contours, key=cv2.contourArea)

    M = cv2.moments(cnt) # retorna um um dicionário de todos os valores de momentos calculados

    area = cv2.contourArea(cnt)
    print("Area:", area)

    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        print("Centroide:", cx, cy)
    else:
        print("O contorno não tem área.")
else:
    print("Não foram encontrados contornos.")


print("Moments: ", M)

'''
Com os momentos, podemos calcular os centroides das imagens
'''
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

'''
Contour area calcula a área de um contorno numa imagem
'''
area = cv2.contourArea(cnt)
print("Area: ", area)


'''
Contour perimeter calcula o perimetro do contorno
'''
perimeter = cv2.arcLength(cnt, True) # true indica se a curva é fechada ou não
print("Perimeter: ", perimeter)

'''
Bounding Rect desenha uma caixa à volta do objeto, permitindo identificar a sua posição e dimensões.
(x, y) são as coordenadas do canto superior esquerdo da imagem
(w, h) são as dimensões da imagem
'''
x, y, w, h = cv2.boundingRect(cnt)
imagem_rect = imagem.copy()


'''
Connected Componentes With Stats procura objetos separados na imagem e calcula informações 
sobre cada um, como a sua área, posição, largura e altura.
'''
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    thresh, connectivity=8
)
for i in range(1, num_labels):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]

    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    area = stats[i, cv2.CC_STAT_AREA]

    cx, cy = centroids[i]

    print("Componente:", i)
    print("Área:", area)
    print("Largura:", w)
    print("Altura:", h)
    print("Centroide:", cx, cy)
    
    cv2.rectangle(imagem_rect, (x, y),(x + w, y + h), (0, 255 ,0) ,2)

print("Número de componentes:", num_labels)

plt.figure(figsize=(15, 8))

plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.imshow(thresh, cmap="gray")
plt.title("Threshold")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.imshow(cv2.cvtColor(imagem_contornos, cv2.COLOR_BGR2RGB))
plt.title("Contornos encontrados")
plt.axis("off")

plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(imagem_rect, cv2.COLOR_BGR2RGB))
plt.title("Retangulo")
plt.axis("off")

plt.tight_layout()
plt.show()
