'''
Exercício 5.2 – Cálculo do Gradiente
    5.2.1 – Determinar o módulo e a fase do gradiente com base num operador diferencial, por exemplo, Sobel;
    5.2.2 – Com base no módulo do gradiente, determine uma imagem de contornos (binarização, exercício 3.2);
Compare os resultados com o algoritmo de Canny.
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 5.2.1
caminho = "./imgs/JonquilFlowers.jpg"
imagem = cv2.imread(caminho)

gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

ksize = (5, 5)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)  
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

magnitude = cv2.magnitude(sobelx, sobely)
fase = cv2.phase(sobelx, sobely, angleInDegrees=True)

mag_vis  = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
fase_vis = (fase / 360 * 255).astype(np.uint8)

# 5.2.2
# limiar manul
T = 100
_, binario = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
binario = binario.astype(np.uint8)

# Otsu
_, bin_otsu = cv2.threshold(mag_vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Canny
edges_canny = cv2.Canny(blur, 50, 150)

plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(mag_vis, cmap="gray")
plt.title("Magnitude")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(fase_vis, cmap="gray")
plt.title("Fase")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(bin_otsu, cmap="gray")
plt.title("Otsu")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(edges_canny, cmap="gray")
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.show()