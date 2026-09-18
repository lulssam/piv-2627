import cv2
import numpy as np
'''
Exercício 2.1 - Cromakey (blue screening);

ImageOut = objectImage * Mask + background * !Mask

OpenCV methods:
* add()
* multiply()
* addWeighted()
'''

falcao = cv2.imread("./imgs/falcon.jpg")
floresta = cv2.imread("./imgs/florest.jpg")
mascara = cv2.imread("./imgs/mask.png")

# mascara = cv2.resize(mascara, (falcao.shape[0], falcao.shape[1]))

print(falcao.shape, floresta.shape, mascara.shape) # confirmar que são todas do mesmo tamanho

# ver um pixel especifico na imagem
print(falcao[300][0]) # [101 236 192]
print(falcao[300][200]) # [149 139 145]

# normalização para ter pixeis entre 0 e 1 
falcao = np.float32(falcao) / 255 
mascara = np.float32(mascara) / 255 

print(falcao[300][0]) # [0.39607844 0.9254902  0.7529412 ]
print(falcao[300][200]) # [0.58431375 0.54509807 0.5686275 ]

parte_falcao = cv2.multiply(falcao, mascara)

print(mascara[300][200]) # [1. 1. 1.] -> tudo a branco
print(mascara[300][0]) # [0. 0. 0.] -> tudo a preto

un_masc = 1 - mascara # inverter mascara

floresta = np.float32(floresta) / 255 # normalização
parte_floresta = cv2.multiply(floresta, un_masc)

# adicionar tudo
final = cv2.add(parte_falcao, parte_floresta)

cv2.imshow("parte falcao", parte_falcao)
cv2.imshow("falcao", falcao)
cv2.imshow("mascara", mascara)
cv2.imshow("un masc", un_masc)
cv2.imshow("parte floresta", parte_floresta)
cv2.imshow("resultado final", final)

cv2.waitKey(0)

cv2.destroyAllWindows()
