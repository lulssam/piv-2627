'''
Exercício 3.4 – Labeling e pseudo-coloração de componentes conexas;
Binariza a imagem, limpa o resultado com operadores morfológicos (open/close)
e usa aux_func/bwLabel + aux_func/psColor para rotular e visualizar os blobs.
'''

import os
import sys
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_func.bwLabel import labeling
from aux_func.psColor import CreateColorMap, Gray2PseudoColor

caminho = "./imgs/formas.png"
img = cv2.imread(caminho)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
Grelha de thresholds para escolher visualmente o valor que melhor separa
a flor/folhas do fundo escuro.
'''
valores_threshold = [80, 100, 120, 140, 160, 180]
threshs = [cv2.threshold(img_gray, v, 255, cv2.THRESH_BINARY)[1] for v in valores_threshold]

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for ax, thresh, valor in zip(axs.flat, threshs, valores_threshold):
    ax.imshow(thresh, cmap="gray")
    ax.set_title(f"Threshold = {valor}")
    ax.axis("off")
plt.tight_layout()
plt.show()

'''
Valor escolhido depois de observar a grelha acima.
'''
valor_escolhido = 160
_, binaria = cv2.threshold(img_gray, valor_escolhido, 255, cv2.THRESH_BINARY)

'''
Limpeza morfológica com kernel retangular 5x5:
* open  (erode -> dilate) remove ruído branco pequeno (salt);
* close (dilate -> erode) preenche buracos pretos pequenos dentro dos blobs (pepper).
'''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
aberta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
limpa = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel)

'''
Labeling das componentes conexas e pseudo-coloração para as distinguir.
'''
num_labels, label_img = labeling(limpa)
color_map = CreateColorMap(num_labels, firstColorBlack=1)
colorida = Gray2PseudoColor(label_img.astype("uint8"), color_map)
colorida_rgb = cv2.cvtColor(colorida, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(limpa, cmap="gray")
axs[0].set_title(f"Binário limpo (threshold={valor_escolhido})")
axs[0].axis("off")

axs[1].imshow(label_img, cmap="nipy_spectral")
axs[1].set_title(f"Labels ({num_labels} regiões)")
axs[1].axis("off")

axs[2].imshow(colorida_rgb)
axs[2].set_title("Pseudo-colorido")
axs[2].axis("off")

plt.tight_layout()
plt.show()
