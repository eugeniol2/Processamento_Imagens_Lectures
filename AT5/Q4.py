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


def turnWhiteIfAnyMatch(covered):
    anyMatch = np.any(covered == WHITE, axis=WINDOW_AXES)

    return np.where(anyMatch, WHITE, BLACK).astype(np.uint8)


def erode(imageNdArray, structuringElement):
    padded = padImage(imageNdArray, structuringElement, WHITE)
    windows = getWindowsUnderElement(padded, structuringElement)
    covered = ignoreZerosOfStructuringElement(windows, structuringElement, WHITE)

    return keepWhiteIfFullMatch(covered)


def dilate(imageNdArray, structuringElement):
    padded = padImage(imageNdArray, structuringElement, BLACK)
    windows = getWindowsUnderElement(padded, structuringElement)
    covered = ignoreZerosOfStructuringElement(windows, structuringElement, BLACK)

    return turnWhiteIfAnyMatch(covered)


def opening(imageNdArray, structuringElement):
    eroded = erode(imageNdArray, structuringElement)

    return dilate(eroded, structuringElement)


def closing(imageNdArray, structuringElement):
    dilated = dilate(imageNdArray, structuringElement)

    return erode(dilated, structuringElement)


def plotImage(rows, cols, position, imageNdArray, title):
    plt.subplot(rows, cols, position)
    plt.imshow(imageNdArray, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")


def plotOriginal(imageNdArray, imageName):
    plt.figure(figsize=(6, 3))
    plotImage(1, 1, 1, imageNdArray, f"Imagem Original - {imageName}")
    plt.tight_layout()


def plotOpeningClosingComparison(original, results):
    plt.figure(figsize=(12, 5))
    for row, (size, opened, closed) in enumerate(results):
        plotImage(len(results), 3, 3 * row + 1, original, "Original")
        plotImage(len(results), 3, 3 * row + 2, opened, f"Abertura {size}x{size}")
        plotImage(len(results), 3, 3 * row + 3, closed, f"Fechamento {size}x{size}")
    plt.tight_layout()


# ================================================================================================

img = readImageFromMemory(
    imgName="fish-16.png",
    imgDirectory="../imagens/imagens/binarias_diversas",
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

opened3 = opening(img, se3)
closed3 = closing(img, se3)

opened5 = opening(img, se5)
closed5 = closing(img, se5)

plotOpeningClosingComparison(img, [(3, opened3, closed3), (5, opened5, closed5)])

plt.show()
