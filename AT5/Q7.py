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


def turnWhiteIfAnyMatch(covered):
    anyMatch = np.any(covered == WHITE, axis=WINDOW_AXES)

    return np.where(anyMatch, WHITE, BLACK).astype(np.uint8)


def dilate(imageNdArray, structuringElement):
    padded = padImage(imageNdArray, structuringElement, BLACK)
    windows = getWindowsUnderElement(padded, structuringElement)
    covered = ignoreZerosOfStructuringElement(windows, structuringElement, BLACK)

    return turnWhiteIfAnyMatch(covered)


def invertImage(imageNdArray):
    return WHITE - imageNdArray


def intersectImages(imageA, imageB):
    bothWhite = (imageA == WHITE) & (imageB == WHITE)

    return np.where(bothWhite, WHITE, BLACK).astype(np.uint8)


def uniteImages(imageA, imageB):
    anyWhite = (imageA == WHITE) | (imageB == WHITE)

    return np.where(anyWhite, WHITE, BLACK).astype(np.uint8)


def defineTargetLocation(shape, seedPoint):
    seed = np.full(shape, BLACK, dtype=np.uint8)
    seed[seedPoint] = WHITE

    return seed


def growInsideHole(previous, structuringElement, complement):
    return intersectImages(dilate(previous, structuringElement), complement)


def fillUntilStable(seed, structuringElement, complement):
    previous = seed
    current = growInsideHole(previous, structuringElement, complement)
    iterations = 1

    while not np.array_equal(current, previous):
        previous = current
        current = growInsideHole(previous, structuringElement, complement)
        iterations += 1

    return current, iterations


def plotImage(rows, cols, position, imageNdArray, title):
    plt.subplot(rows, cols, position)
    plt.imshow(imageNdArray, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")


def markSeedPoint(seedPoint):
    plt.scatter(
        seedPoint[1], seedPoint[0], s=200, facecolors="none", edgecolors="red"
    )


def plotOriginal(imageNdArray, imageName, seedPoint):
    plt.figure(figsize=(6, 3))
    plotImage(1, 1, 1, imageNdArray, f"Imagem Original - {imageName}")
    markSeedPoint(seedPoint)
    plt.tight_layout()


def plotHoleFillingResults(original, filledRegion, filledImage, seedPoint):
    plt.figure(figsize=(14, 3.5))
    plotImage(1, 3, 1, original, "Imagem Original")
    markSeedPoint(seedPoint)
    plotImage(1, 3, 2, filledRegion, "Região Encontrada")
    plotImage(1, 3, 3, filledImage, "Após o Preenchimento")
    plt.tight_layout()


# ================================================================================================

img = readImageFromMemory(
    imgName="fish-16.png",
    imgDirectory="../imagens/imagens/binarias_diversas",
    readMode=cv2.IMREAD_GRAYSCALE,
)

seedPoint = (53, 117)
print(f"Ponto no interior do buraco: linha {seedPoint[0]}, coluna {seedPoint[1]}")

plotOriginal(img, "fish-16.png", seedPoint)

seed = defineTargetLocation(img.shape, seedPoint)

complement = invertImage(img)

se = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

filledRegion, iterations = fillUntilStable(seed, se, complement)
print(f"Iterações até estabilizar: {iterations}")

filledImage = uniteImages(img, filledRegion)

plotHoleFillingResults(img, filledRegion, filledImage, seedPoint)

plt.show()
