import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import readImageFromMemory

GRAY_LEVELS = 256


def getDimensions(imageNdArray):
    height, width = imageNdArray.shape

    return height, width


def countTotalPixels(height, width):
    return height * width


def computeHistogram(imageNdArray):
    histogram = np.zeros(GRAY_LEVELS, dtype=np.int64)

    for pixelValue in imageNdArray.ravel():
        histogram[pixelValue] += 1

    return histogram


def normalizeHistogram(histogram, totalPixels):
    return histogram / totalPixels


def computeCdf(normalizedHistogram):
    return np.cumsum(normalizedHistogram)


def computeTransformation(cdf):
    transformation = np.round((GRAY_LEVELS - 1) * cdf)

    return np.uint8(transformation)


def applyTransformation(imageNdArray, transformation):
    return transformation[imageNdArray]


def plotHistogram(position, histogram, title):
    plt.subplot(2, 3, position)
    plt.bar(np.arange(GRAY_LEVELS), histogram, width=1)
    plt.xlim(0, GRAY_LEVELS - 1)
    plt.title(title)
    plt.xlabel("Nível de intensidade")
    plt.ylabel("Quantidade de pixels")


def plotImage(position, imageNdArray, title):
    plt.subplot(2, 3, position)
    plt.imshow(imageNdArray, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")


def plotTransformation(position, transformation):
    plt.subplot(2, 3, position)
    plt.plot(np.arange(GRAY_LEVELS), transformation)
    plt.xlim(0, GRAY_LEVELS - 1)
    plt.ylim(0, GRAY_LEVELS - 1)
    plt.title("Função de Transformação $s_k$")
    plt.xlabel("Intensidade original $r_k$")
    plt.ylabel("Intensidade equalizada $s_k$")


def plotEqualizationResults(
    original, originalHistogram, transformation, equalized, equalizedHistogram
):
    plt.figure(figsize=(15, 8))
    plotImage(1, original, "Imagem Original")
    plotHistogram(2, originalHistogram, "Histograma Original")
    plotTransformation(3, transformation)
    plotImage(4, equalized, "Imagem Equalizada")
    plotHistogram(5, equalizedHistogram, "Histograma Equalizado")
    plt.tight_layout()


# ================================================================================================

img = readImageFromMemory(
    imgName="cameraman_1.png",
    imgDirectory="../imagens/imagens/histograma",
    readMode=cv2.IMREAD_GRAYSCALE,
)

height, width = getDimensions(img)

totalPixels = countTotalPixels(height, width)

hist = computeHistogram(img)

normalizedHist = normalizeHistogram(hist, totalPixels)

cdf = computeCdf(normalizedHist)

transformation = computeTransformation(cdf)

equalizedImg = applyTransformation(img, transformation)

equalizedHist = computeHistogram(equalizedImg)

plotEqualizationResults(img, hist, transformation, equalizedImg, equalizedHist)

plt.show()
