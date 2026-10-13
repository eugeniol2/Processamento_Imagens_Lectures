import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import readImageFromMemory

WHITE = 255
BLACK = 0
WINDOW_AXES = (-2, -1)


def padImage(imageNdArray, structuringElement, fillValue):
    radius = structuringElement.shape[0] // 2

    return np.pad(imageNdArray, radius, mode="constant", constant_values=fillValue)


def getWindowsUnderElement(paddedImage, structuringElement):
    return np.lib.stride_tricks.sliding_window_view(
        paddedImage, structuringElement.shape
    )


def ignoreZerosOfStructuringElement(windows, structuringElement, neutralValue):
    return np.where(structuringElement == 1, windows, neutralValue)


def keepWhiteIfFullMatch(covered):
    fullMatch = np.all(covered == WHITE, axis=WINDOW_AXES)

    return np.where(fullMatch, WHITE, BLACK).astype(np.uint8)


def erode(imageNdArray, structuringElement):
    padded = padImage(imageNdArray, structuringElement, WHITE)
    windows = getWindowsUnderElement(padded, structuringElement)
    covered = ignoreZerosOfStructuringElement(windows, structuringElement, WHITE)

    return keepWhiteIfFullMatch(covered)


def subtractImages(original, eroded):
    return original - eroded


def plotImage(rows, cols, position, imageNdArray, title):
    plt.subplot(rows, cols, position)
    plt.imshow(imageNdArray, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")


def plotOriginal(imageNdArray, imageName):
    plt.figure(figsize=(6, 3))
    plotImage(1, 1, 1, imageNdArray, f"Imagem Original - {imageName}")
    plt.tight_layout()


def plotBorderComparison(original, results):
    plt.figure(figsize=(12, 7))
    for row, (size, eroded, border) in enumerate(results):
        plotImage(len(results), 3, 3 * row + 1, original, "Original")
        plotImage(len(results), 3, 3 * row + 2, eroded, f"Erosão {size}x{size}")
        plotImage(len(results), 3, 3 * row + 3, border, f"Bordas {size}x{size}")
    plt.tight_layout()


# ================================================================================================

img = readImageFromMemory(
    imgName="fish-16.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

plotOriginal(img, imageName="fish-16.png")

se3 = np.array(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)

se5 = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

se9 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

eroded3 = erode(img, se3)
border3 = subtractImages(img, eroded3)

eroded5 = erode(img, se5)
border5 = subtractImages(img, eroded5)

eroded9 = erode(img, se9)
border9 = subtractImages(img, eroded9)

plotBorderComparison(
    img,
    [(3, eroded3, border3), (5, eroded5, border5), (9, eroded9, border9)],
)

plt.show()
