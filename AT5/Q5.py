import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import readImageFromMemory

WHITE = 255
BLACK = 0
GRAY = 128
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


def invertImage(imageNdArray):
    return WHITE - imageNdArray


def keepWhereBothMatch(foregroundMatch, backgroundMatch):
    bothMatch = (foregroundMatch == WHITE) & (backgroundMatch == WHITE)

    return np.where(bothMatch, WHITE, BLACK).astype(np.uint8)


def hitOrMiss(imageNdArray, foregroundElement, backgroundElement):
    foregroundMatch = erode(imageNdArray, foregroundElement)
    backgroundMatch = erode(invertImage(imageNdArray), backgroundElement)

    return keepWhereBothMatch(foregroundMatch, backgroundMatch)


def getMatchPositions(hitOrMissResult):
    return np.argwhere(hitOrMissResult == WHITE)


def printMatchPositions(patternName, positions):
    print(f"{patternName}: {len(positions)} ocorrência(s)")
    for row, col in positions:
        print(f"  linha {row}, coluna {col}")


def buildPatternPreview(foregroundElement, backgroundElement):
    preview = np.full(foregroundElement.shape, GRAY, dtype=np.uint8)
    preview[foregroundElement == 1] = WHITE
    preview[backgroundElement == 1] = BLACK

    return preview


def plotImage(rows, cols, position, imageNdArray, title):
    plt.subplot(rows, cols, position)
    plt.imshow(imageNdArray, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")


def markPositions(positions):
    plt.scatter(
        positions[:, 1], positions[:, 0], s=150, facecolors="none", edgecolors="red"
    )


def showFrame():
    plt.axis("on")
    plt.xticks([])
    plt.yticks([])


def plotOriginal(imageNdArray, imageName):
    plt.figure(figsize=(6, 3))
    plotImage(1, 1, 1, imageNdArray, f"Imagem Original - {imageName}")
    plt.tight_layout()


def plotHitOrMissResults(original, patternPreview, hitOrMissResult, positions, patternName):
    plt.figure(figsize=(14, 3.5))
    plotImage(1, 3, 1, patternPreview, f"Padrão: {patternName}")
    showFrame()
    plotImage(1, 3, 2, hitOrMissResult, "Resultado Hit-or-Miss")
    markPositions(positions)
    plotImage(1, 3, 3, original, f"Posições encontradas ({len(positions)})")
    markPositions(positions)
    plt.tight_layout()


# ================================================================================================

img = readImageFromMemory(
    imgName="fish-16.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

plotOriginal(img, imageName="fish-16.png")

crossForeground = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

crossBackground = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

crossResult = hitOrMiss(img, crossForeground, crossBackground)
crossPositions = getMatchPositions(crossResult)
printMatchPositions("Cruz branca isolada", crossPositions)

plotHitOrMissResults(
    img,
    buildPatternPreview(crossForeground, crossBackground),
    crossResult,
    crossPositions,
    "cruz branca isolada",
)


diamondHoleForeground = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

diamondHoleBackground = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

diamondResult = hitOrMiss(img, diamondHoleForeground, diamondHoleBackground)
diamondPositions = getMatchPositions(diamondResult)
printMatchPositions("Buraco preto em losango", diamondPositions)

plotHitOrMissResults(
    img,
    buildPatternPreview(diamondHoleForeground, diamondHoleBackground),
    diamondResult,
    diamondPositions,
    "buraco em losango",
)

plt.show()
