import cv2
import numpy as np

from utils import readImageFromMemory, saveImage


# Exemplo usado nos comentários abaixo: uma imagem 2x3 ampliada por 2
#
#   imagem = [[10, 20, 30],       resultado ampliado = [[10, 20, 20, 30, 30, 30],
#             [40, 50, 60]]                             [40, 50, 50, 60, 60, 60],
#                                                       [40, 50, 50, 60, 60, 60],
#                                                       [40, 50, 50, 60, 60, 60]]


def nearestSourceIndices(originalSize, upscaledSize):
    # Só entram tamanhos, nunca valores de pixel. Para as linhas do exemplo:
    # originalSize = 2 (height), upscaledSize = 4 (height * factor)

    # todas as posições i da imagem ampliada, de uma vez
    #   np.arange(4) = [0, 1, 2, 3]
    outputIndices = np.arange(upscaledSize)

    # x = i . n / m, onde cada posição cai na imagem original
    #   [0, 1, 2, 3] * 2 = [0, 2, 4, 6]
    #   [0, 2, 4, 6] / 4 = [0.0, 0.5, 1.0, 1.5]
    realPositions = outputIndices * originalSize / upscaledSize

    # parte decimal de cada posição
    #   [0.0, 0.5, 1.0, 1.5] - [0, 0, 1, 1] = [0.0, 0.5, 0.0, 0.5]
    fraction = realPositions - np.floor(realPositions)

    # x' = round(x): fração < 0.5 desce, senão sobe
    #   [0.0, 0.5, 1.0, 1.5] -> [0, 1, 1, 2]
    rounded = np.where(
        fraction < 0.5, np.floor(realPositions), np.ceil(realPositions)
    ).astype(int)

    # a linha 2 não existe (a original só tem as linhas 0 e 1), então é
    # limitada à última
    #   minimum([0, 1, 1, 2], 1) = [0, 1, 1, 1]
    return np.minimum(rounded, originalSize - 1)


def upscaleNearestNeighbor(imageNdArray, factor):
    # height = 2, width = 3 -> a imagem ampliada terá 4 x 6
    height, width = imageNdArray.shape

    # qual linha da original cada linha da ampliada copia
    #   nearestSourceIndices(2, 4) = [0, 1, 1, 1]
    sourceRows = nearestSourceIndices(height, height * factor)

    # qual coluna da original cada coluna da ampliada copia
    #   nearestSourceIndices(3, 6):
    #   [0, 1, 2, 3, 4, 5] * 3 / 6 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    #   arredondado                = [0, 1, 1, 2, 2, 3]
    #   limitado a 2               = [0, 1, 1, 2, 2, 2]
    sourceColumns = nearestSourceIndices(width, width * factor)

    # sourceRows[:, None] fica em pé (4x1), sourceColumns[None, :] fica deitado
    # (1x6), e o broadcasting cruza os dois em duas grades 4x6 de origem:
    #
    #   linhas de origem:     colunas de origem:
    #   [[0 0 0 0 0 0]        [[0 1 1 2 2 2]
    #    [1 1 1 1 1 1]         [0 1 1 2 2 2]
    #    [1 1 1 1 1 1]         [0 1 1 2 2 2]
    #    [1 1 1 1 1 1]]        [0 1 1 2 2 2]]
    #
    # o pixel (1, 3) da saída (4x6, colunas 0 a 5) lê o pixel (1, 2) da
    # original (2x3, colunas 0 a 2), que vale 60
    # os valores dos pixels só são copiados aqui, nunca calculados
    return imageNdArray[sourceRows[:, None], sourceColumns[None, :]]


# ================================================================================================

baboonImg = readImageFromMemory(
    imgName="baboon_monocromatica.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

baboonUpscaled2 = upscaleNearestNeighbor(baboonImg, factor=2)
baboonUpscaled4 = upscaleNearestNeighbor(baboonImg, factor=4)

saveImage(imageNdArray=baboonUpscaled2, name="baboon_upscaled_2x", imgType="png")
saveImage(imageNdArray=baboonUpscaled4, name="baboon_upscaled_4x", imgType="png")

