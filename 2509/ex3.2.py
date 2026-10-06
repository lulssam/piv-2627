'''
Exercício 3.2 – Binarização de uma imagem;
OpenCV methods: 
* threshold()
* adaptiveThreshold()
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

caminho = "./imgs/lily-lotus-flowers.jpg"
img = cv2.imread(caminho)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

'''
Binary threshold

Se um pixel estiver acima do threshold, é lhe atribuido o valor máximo (255, branco),
caso contrário, fica a 0 (preto).

Útil para separar objetos de um fundo escuro.
'''

_, thresh_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

'''
Binary threshold inverted

É o oposto do binario.
'''
_, thresh_binary_inv = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)

'''
Truncated Threshold

Pixeis acima do threshold assumem o valor do threshold. Pixeis abaixo, 
continuam igual

Limita a luminosidade máxima, util para nivelar parte overexposed.
'''

_, thresh_trunc = cv2.threshold(img_gray, 120, 255, cv2.THRESH_TRUNC)

'''
To Zero Threshold

Extrai as partes mais iluminadas da imagem enquanto baixa as mais escuras
'''

_, to_zero = cv2.threshold(img_gray, 120, 255, cv2.THRESH_TOZERO)

'''
To Zero Inverted

Pixeis acima do threshold ficam a 0. Pixeis abaixo, mantém o seu valor
original.

Mantém detalhes escuros enquanto mascara as mais brilhantes.
'''

_, to_zero_inv = cv2.threshold(img_gray, 120, 255, cv2.THRESH_TOZERO_INV)


'''
Adaptive Thresholding aborda uma das maiores limitações do threshold
normal, a sua inabilidade de tratar de imagens com variações de luz.

Invez de usar um valor global de threshold para toda a imagem, este método
calcula o threshold para regiões mais pequenas a volta de cada pixel.
'''

'''
Adaptive Mean Thresholding

O threshold de cada pixel é computado como a media do bloco a volta, 
menos a contante.

Melhor para imagens com ruído consistente.

cv2.adaptiveThreshold() leva os seguintes argumentos:
* imagem de 1 canal: img_gray
* maxValue: 255
* tipo de threshold: cv2.ADAPTIVE_THRESH_MEAN_C
* blocksize: 199: tamanho da vizinhança usada para calcular o threshold local
    em torno de cada pixel. Tem de ser impar.
* C: 5: constante subtraída da média calculada. Serva para ajustar a sensabilidade
    do threshold
'''

thresh_mean = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    27, 5
)

thresh_mean_inv = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    11, 5
)

'''
Adaptive Gaussian Thresholding

O threshold de cada pixel vem de uma soma ponderada dos pixeis vizinhos
(Gaussian Window), menos a constante.

Mais efetivo do que a média em regiões com variações graduais de intensidade
'''

thresh_gauss = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 5
)

thresh_gauss_inv = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 5
)

imagens = [img_rgb, img_gray, thresh_binary, thresh_binary_inv, thresh_trunc, to_zero, to_zero_inv, thresh_mean, thresh_mean_inv, thresh_gauss, thresh_gauss_inv]
titulos = ["Original", "Escala de Cinzentos", "Binário", "Binário Invertido", "Truncado", "To Zero", "To Zero Invertido", "Threshold Média", "Threshold Média Invertido", "Threshold Gaussiano", "Threshold Gaussiano Invertido"]

fig, axs = plt.subplots(3, 4, figsize=(14, 9))

for ax, imagem, titulo in zip(axs.flat, imagens, titulos):
    ax.imshow(imagem, cmap="gray")
    ax.set_title(titulo)
    ax.axis("off")

axs.flat[-1].axis("off") 

plt.tight_layout()
plt.show()