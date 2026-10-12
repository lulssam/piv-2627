import matplotlib.pyplot as plt
import cv2
import numpy as np

# 1. leitura de uma imagem
caminho = "./TP1/treino/lego01.jpg"
img = cv2.imread(caminho)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# split do hsv
h, s, v = cv2.split(img_hsv)

# binarização
thresh = 150
_, binaria = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

# histogramas
histH = cv2.calcHist([h], [0], None, [180], [0, 180])
histS = cv2.calcHist([s], [0], None, [256], [0, 256])

# picos de cada histograma
maxH = np.argmax(histH)
maxS = np.argmax(histS)
print("MAX H: ",maxH)
print("MAX S: ",maxS)

# margens
margem_h = 10
margem_s = 40

h_min = np.clip(maxH - margem_h, 0, 179)
h_max = np.clip(maxH + margem_h, 0, 179)
s_min = np.clip(maxS - margem_s, 0, 255)
s_max = np.clip(maxS + margem_s, 0, 255)

# limites
minimo = np.array([h_min, s_min, 0], dtype=np.uint8)
maximo = np.array([h_max, s_max, 255], dtype=np.uint8)

# mascara da mesa
mascara_mesa = cv2.inRange(img_hsv, minimo, maximo)

# mascara da peça -> o contrário da mesa
mascara_peca = cv2.bitwise_not(mascara_mesa)

# melhoramento da imagem
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
aberta = cv2.morphologyEx(mascara_peca, cv2.MORPH_OPEN, kernel=kernel)
limpa = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel=kernel)


imagens = [img_rgb, img_gray, mascara_mesa, mascara_peca, aberta, limpa]
titulos = ["Original", "Escala de Cinzentos", "Mascara Mesa", "Mascara Peca", "Aberta", "Limpa"]
fig, axs = plt.subplots(2, 4, figsize=(14, 9))

for ax, imagem, titulo in zip(axs.flat, imagens, titulos):
    ax.imshow(imagem, cmap="gray")
    ax.set_title(titulo)
    ax.axis("off")

ax = axs.flat[6]
ax.plot(histH, color="orange")
ax.set_xlim([0, 180])
ax.set_title("Histograma H")

ax = axs.flat[7]
ax.plot(histS, color="purple")
ax.set_xlim([0, 256])
ax.set_title("Histograma S")

plt.tight_layout()
plt.show()