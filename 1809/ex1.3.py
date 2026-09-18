'''
Exercício 1.3 - Redimensionamento de imagens;
OpenCV method:
* resize()
'''

import cv2

caminho_img = "./imgs/JonquilFlowers.jpg" # caminho da imagem
janela = cv2.imread(caminho_img) # ler imagem

cv2.imshow("Imagem Original", janela) # mostrar imagem original

resized_absoluto = cv2.resize(janela, (1280, 720)) # trocar dimensões
cv2.imshow("Resized Absoluto", resized_absoluto) # mostrar imagem

resize_scale = cv2.resize(janela, None, fx=0.5, fy=0.5) # metade das dimensões
cv2.imshow("Resized Scale", resize_scale) # mostrar imagem

while cv2.waitKey(30) & 0xFF != ord("q"):
    pass

cv2.destroyAllWindows()


'''
há dois tipos de resize, absoluto onde damos uma dimensão à escolha
e o scale que é proporcional
'''
