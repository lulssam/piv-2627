'''
Exercicio 5.1 - Deteção de contornos;
OpenCV methods: 
* cvtColor(); 
* Sobel(); 
* Canny(); 
* Filter2D() -> já usado no exercicio 2.2 
* Laplacian();
* convertScaleAbs()
'''
import cv2
import matplotlib.pyplot as plt

caminho = "./imgs/JonquilFlowers.jpg"
imagem = cv2.imread(caminho)

gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

'''
Sobel: operador diferencial de 1ª ordem. O kernel combina suavização
([1 2 1], aproximação gaussiana) na direção perpendicular com
diferenciação ([-1 0 1]) na direção da derivada -> menos sensível a ruído.

sobelx (dx=1): derivada em x -> realça contornos VERTICAIS
sobely (dy=1): derivada em y -> realça contornos HORIZONTAIS

cv2.magnitude() -> módulo do gradiente |∇I| = sqrt(Gx² + Gy²),
intensidade do contorno em cada pixel.
'''
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # vertical
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) # horizontal

magnitude = cv2.magnitude(sobelx, sobely) # calcular magnitude
magnitude = cv2.convertScaleAbs(magnitude) # converter para 8 bits

'''
Canny: detetor de contornos multi-etapa, desenhado para cumprir 3 critérios:
  - boa deteção (poucos falsos positivos/negativos)
  - boa localização (contorno no sítio certo)
  - resposta única (contornos com 1 pixel de espessura)

Etapas:
  1. Suavização gaussiana (σ) para reduzir ruído.
     NOTA: o cv2.Canny NÃO faz este passo -> fazer antes com GaussianBlur.
  2. Gradiente (Sobel): módulo e direção em cada pixel.
  3. Supressão de não-máximos: mantém só os pixels que são máximo local
     do módulo ao longo da direção do gradiente -> contornos finos.
  4. Limiarização por histerese (T_low, T_high):
       |∇I| >= T_high           -> contorno forte (aceite)
       |∇I| <  T_low            -> rejeitado
       T_low <= |∇I| < T_high   -> aceite só se ligado a um contorno forte
     (rácio típico T_high:T_low entre 2:1 e 3:1)

Resultado: imagem binária (0/255) com contornos finos e contínuos.

edges = cv2.Canny(src, threshold1, threshold2, apertureSize=3, L2gradient=False)
  L2gradient=False -> usa |Gx|+|Gy| (aproximação mais rápida da norma L2)
'''
t_lower = 50
t_high = 150
edges = cv2.Canny(blur, t_lower, t_high)

'''
Canny com Aperture Size
Podemos aumentar o apertura size se queremos mais detalhes.
'''
edge_as = cv2.Canny(blur, t_lower, t_high, apertureSize=5)

'''
Canny com L2Gradient
Parametro booleano que especifica se queremos calcular a equação do gradiente habitual ou utilizar o algoritmo L2Gradient
'''

edge_l2 = cv2.Canny(blur, t_lower, t_high, L2gradient=True)


'''
Laplace
'''
ddepth = cv2.CV_16S
ksize = 3
dst = cv2.Laplacian(blur, ddepth, ksize)
dst_8bit = cv2.convertScaleAbs(dst)

plt.figure(figsize=(15, 8))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Escala Cinzentos")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(sobelx, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(sobely, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(magnitude, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(edges, cmap="gray")
plt.title("Canny")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(edge_as, cmap="gray")
plt.title("Canny Aperture Size")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(edge_l2, cmap="gray")
plt.title("Canny L2 Gradient")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(edge_l2, cmap="gray")
plt.title("Canny L2 Gradient")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.imshow(dst_8bit, cmap="gray")
plt.title("Laplacian")
plt.axis("off")


plt.tight_layout()
plt.show()



