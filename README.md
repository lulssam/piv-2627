# PIV 2026/27 — Processamento de Imagem e Visão

Exercícios das aulas e trabalhos práticos da unidade curricular de **Processamento de Imagem e Visão** da Licenciatura em Engenharia Informática e Multimédia (LEIM) do ISEL, ano letivo 2026/2027.

Tudo em Python com OpenCV, NumPy e Matplotlib.

## Estrutura

```
piv-2627/
├── 1809/          # Aula de 18/09 — leitura, vídeo, resize, chroma key, filtros, rotação
├── 2509/          # Aula de 25/09 — histogramas, binarização, morfologia, labeling
├── 1311/          # Aula de 13/11 — extração de características, contornos, gradiente
├── aux_func/      # Funções auxiliares (labeling e pseudo-cor)
├── imgs/          # Imagens usadas nos exercícios
├── videos/        # Vídeo de teste (vtest.mp4)
└── TP1/           # 1º Trabalho Prático — contagem e classificação de peças de Lego
```

As pastas das aulas têm o nome da data da aula (`DDMM`).

## Exercícios

| Ficheiro | Tema | Principais métodos OpenCV |
|---|---|---|
| `1809/ex1.1.py` | Leitura e visualização de uma imagem; conversão para cinzentos | `imread`, `imshow`, `cvtColor`, `waitKey` |
| `1809/ex1.2.py` | Leitura de vídeo e de webcam | `VideoCapture`, `get`, `read`, `release` |
| `1809/ex1.3.py` | Redimensionamento absoluto e por escala | `resize` |
| `1809/ex2.1.py` | Chroma key: `out = objeto·máscara + fundo·(1 − máscara)` | `multiply`, `add` |
| `1809/ex2.2.py` | Filtros de média, mediana e gaussiano sobre ruído sal-e-pimenta; kernels de deteção de arestas, *sharpen* e *box* | `blur`, `medianBlur`, `GaussianBlur`, `getGaussianKernel`, `filter2D` |
| `1809/ex2.3.py` | Transformações geométricas (rotação) | `getRotationMatrix2D`, `warpAffine` |
| `2509/ex3.1.py` | Histograma por canal (B, G, R) | `calcHist` |
| `2509/ex3.2.py` | Binarização global (binary, trunc, to zero e inversos) e adaptativa (média e gaussiana) | `threshold`, `adaptiveThreshold` |
| `2509/ex3.3.py` | Operadores morfológicos (erosão) | `erode` |
| `2509/ex3.4_labeling.py` | Binarização, limpeza com open/close, labeling de componentes conexas e pseudo-coloração | `morphologyEx`, `getStructuringElement` + `aux_func` |
| `1311/ex3.5.py` | Extração de características: área, perímetro, centroide, *bounding box*, componentes conexas | `findContours`, `moments`, `contourArea`, `arcLength`, `boundingRect`, `connectedComponentsWithStats` |
| `1311/ex5.1.py` | Deteção de contornos: Sobel (X, Y, magnitude), Canny (com `apertureSize` e `L2gradient`), Laplaciano | `Sobel`, `magnitude`, `Canny`, `Laplacian`, `convertScaleAbs` |
| `1311/ex5.2.py` | Módulo e fase do gradiente; contornos por binarização (Otsu) do módulo vs. Canny | `Sobel`, `magnitude`, `phase`, `threshold` |

### `aux_func/`

- **`bwLabel.labeling(bwImage)`** — recebe uma imagem binária e devolve `(n_labels, imagem_rotulada)`, em que cada blob tem um valor inteiro diferente. Usa `findContours` com `RETR_CCOMP` e percorre só os contornos exteriores (os buracos ficam preenchidos). Compatível com OpenCV 2, 3 e 4.
- **`psColor.CreateColorMap(n, firstColorBlack=0)`** — gera uma tabela de `n` cores distintas a partir de três sinusoides desfasadas.
- **`psColor.Gray2PseudoColor(grayImage, colorMap)`** — substitui cada label pela cor correspondente da tabela.

## TP1 — Contagem e classificação de peças de Lego retangulares

`TP1/TP1_52D_A48628_A50799/` contém o notebook, o relatório em PDF e as imagens de `treino/` (10) e `validacao/` (4).

Pipeline:

1. **Binarização** — desfoque gaussiano do canal vermelho e *threshold* fixo (85).
2. **Melhoramento** — fecho (elipse 5×5), erosão (elipse 15×15, 3 iterações) e dilatação (elipse 11×11, 3 iterações) para remover ruído e separar peças.
3. **Contornos** — `findContours` com `RETR_EXTERNAL`; contornos com área < 1000 px são ignorados.
4. **Classificação** — retângulo de área mínima (`minAreaRect`) e razão de aspeto (lado maior / lado menor):

   | Razão de aspeto | Peça |
   |---|---|
   | < 1,25 | 2×2 |
   | 1,82 – 2,42 | 2×4 |
   | 2,42 – 3,57 | 2×6 |
   | > 3,57 | 2×8 |
   | outro, ou lado < 60 px | Inválida |

5. **Saída** — cada peça é anotada na imagem e a contagem por tipo é escrita numa faixa por baixo. As imagens são guardadas em `output_images/`.

Autores: Filipe Mendes (48628) e Maria Luísa Melo e Sampaio (50799), turma 52D. Docente: Tiago Gonçalves.

## Como correr

Requisitos: Python 3.10+.

```bash
pip install opencv-python numpy matplotlib jupyter
```

Os exercícios usam caminhos relativos (`./imgs/...`, `./videos/...`), por isso têm de ser corridos **a partir da raiz do repositório**:

```bash
python 1809/ex1.1.py
python 2509/ex3.4_labeling.py
```

Nas janelas do OpenCV, carrega em `q` (ou em qualquer tecla, nos exercícios com `waitKey(0)`) para fechar. O `ex1.2.py` abre a webcam depois do vídeo.

O TP1 lê de `./validacao/`, por isso o notebook tem de ser aberto a partir da pasta dele:

```bash
cd TP1/TP1_52D_A48628_A50799
jupyter notebook TP1_52D_A48628_A50799.ipynb
```

Para correr sobre as imagens de treino, muda `images_dir` para `'./treino/'`.
