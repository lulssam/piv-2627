'''
Exercício 2.2 - Filtragem de média e mediana
OpenCV methods:
* blur()
* medianBlur()
* GaussianBlur()
* cv.getGaussianKernel()
* filter2D()
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

caminho = "./imgs/BrainMRI_Axial.jpg"

original = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

# adicionar ruído branco
ruido = original.copy()
random = np.random.rand(original.shape[0], original.shape[1])

ruido[random < 0.05] = 0
ruido[random > 0.95] = 255

ksize = (7, 7) # impar e positivo

edgeDetect = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])

sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

box = 1/9 * np.ones((3, 3))

blur = cv2.blur(ruido, ksize=ksize) # blur
median = cv2.medianBlur(ruido, 5) # tem que ser impar
gaussian = cv2.GaussianBlur(ruido, (11, 11), 0)

gaussainKernel = cv2.getGaussianKernel(3, 1) # commonly used for smoothing and blurring operations in image processing
print(gaussainKernel)

filter2d = cv2.filter2D(ruido, ddepth= -1,kernel=gaussainKernel) # convolve an image with the kernel (typically a 2d matrix) to apply a filter on the images.
filter2dED = cv2.filter2D(original, ddepth=-1, kernel=edgeDetect)
filter2dSharpen = cv2.filter2D(original, ddepth=-1, kernel=sharpen)
filter2dBox = cv2.filter2D(original, ddepth=-1, kernel=box)

# plots
imagens = [original, ruido, blur, median, gaussian, filter2d, filter2dED, filter2dSharpen, filter2dBox]
titulos = ["Original", "Ruido", "Blur 7x7", "Median 5", "Gaussian 11x11", "Filter2D", "EdgeDetect", "Sharpen", "Box"]

fig, axs = plt.subplots(3, 3, figsize=(12, 8))

for ax, img, titulo in zip(axs.flat, imagens, titulos):
    ax.imshow(img, cmap="gray")
    ax.set_title(titulo)
    ax.axis("off")

plt.tight_layout()
plt.show()