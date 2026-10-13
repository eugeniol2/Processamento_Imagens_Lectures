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


def plotImageAndHistograms(
    imageNdArray, histogram, normalizedHistogram, normalizedYMax=None
):
    levels = np.arange(GRAY_LEVELS)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(imageNdArray, cmap="gray", vmin=0, vmax=255)
    plt.title("Imagem Original")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.bar(levels, histogram, width=1)
    plt.xlim(0, GRAY_LEVELS - 1)
    plt.title("Histograma")
    plt.xlabel("Nível de intensidade")
    plt.ylabel("Quantidade de pixels")
    plt.subplot(1, 3, 3)
    plt.bar(levels, normalizedHistogram, width=1)
    plt.xlim(0, GRAY_LEVELS - 1)
    plt.ylim(0, normalizedYMax)
    plt.title("Histograma Normalizado")
    plt.xlabel("Nível de intensidade")
    plt.ylabel("Probabilidade")
    plt.tight_layout()


def getIntensityRange(histogram):
    usedLevels = np.nonzero(histogram)[0]

    return usedLevels[0], usedLevels[-1]


def computeMeanIntensity(normalizedHistogram):
    return np.sum(np.arange(GRAY_LEVELS) * normalizedHistogram)


def getSharedYMax(normalizedA, normalizedB):
    return max(normalizedA.max(), normalizedB.max()) * 1.05


def printHistogramSummary(name, histogram, normalizedHistogram):
    low, high = getIntensityRange(histogram)
    mean = computeMeanIntensity(normalizedHistogram)
    print(f"{name}: faixa {low}-{high} ({high - low + 1} níveis) | média {mean:.1f}")


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

plotImageAndHistograms(img, hist, normalizedHist)


otherImg = readImageFromMemory(
    imgName="cameraman_4.png",
    imgDirectory="../imagens/imagens/histograma",
    readMode=cv2.IMREAD_GRAYSCALE,
)

otherHeight, otherWidth = getDimensions(otherImg)

otherTotalPixels = countTotalPixels(otherHeight, otherWidth)

otherHist = computeHistogram(otherImg)

otherNormalizedHist = normalizeHistogram(otherHist, otherTotalPixels)

printHistogramSummary("cameraman_1", hist, normalizedHist)
printHistogramSummary("cameraman_4", otherHist, otherNormalizedHist)

sharedYMax = getSharedYMax(normalizedHist, otherNormalizedHist)

plotImageAndHistograms(img, hist, normalizedHist, sharedYMax)
plotImageAndHistograms(otherImg, otherHist, otherNormalizedHist, sharedYMax)

plt.show()
