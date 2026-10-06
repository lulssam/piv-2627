'''
Exercício 3.1 – Histograma de uma imagem;
OpenCV method: calcHist();
Matplotlib (python 2D plotting library) method: bar(); show();
'''
import cv2
import matplotlib.pyplot as plt

caminho = "./imgs/lily-lotus-flowers.jpg"
img = cv2.imread(caminho)


histBlue = cv2.calcHist([img],[0],None,[256],[0,256])
histGreen = cv2.calcHist([img],[1],None,[256],[0,256])
histRed = cv2.calcHist([img],[2],None,[256],[0,256])

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title("Imagem")
axs[0, 0].axis("off")

axs[0, 1].bar(range(256), histBlue.ravel(), color="blue")
axs[0, 1].set_title("Canal Azul")

axs[1, 0].bar(range(256), histGreen.ravel(), color="green")
axs[1, 0].set_title("Canal Verde")

axs[1, 1].bar(range(256), histRed.ravel(), color="red")
axs[1, 1].set_title("Canal Encarnado")
plt.tight_layout()

plt.show()