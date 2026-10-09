import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import readImageFromMemory


def convertToFloat(imageNdArray):
    return np.float32(imageNdArray)


def computeFourierTransform(imageNdArray):
    return np.fft.fft2(imageNdArray)


def centerSpectrum(f):
    return np.fft.fftshift(f)


def computeMagnitudeSpectrum(fshift):
    return 20 * np.log(np.abs(fshift) + 1)


def computePhaseSpectrum(fshift):
    return np.angle(fshift)


def uncenterSpectrum(fshift):
    return np.fft.ifftshift(fshift)


def buildLowPassMask(imageNdArray, r):
    rows, cols = imageNdArray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (i - crow) ** 2 + (j - ccol) ** 2 <= r**2:
                mask[i, j] = 1

    return mask


def buildHighPassMask(imageNdArray, r):
    rows, cols = imageNdArray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (i - crow) ** 2 + (j - ccol) ** 2 <= r**2:
                mask[i, j] = 0

    return mask


def applyMask(fshift, mask):
    return fshift * mask


def computeInverseFourierTransform(fshiftFiltered):
    fIshift = uncenterSpectrum(fshiftFiltered)
    imgBack = np.fft.ifft2(fIshift)
    imgBack = np.abs(imgBack)

    return imgBack


def plotFilteredImage(imgBack):
    plt.imshow(imgBack, cmap="gray")
    plt.title("Imagem Filtrada")
    plt.show()


def computeAbsoluteDifference(original, reconstructed):
    return np.abs(original - reconstructed)


def convertToGrayLevels(imageNdArray):
    return np.round(imageNdArray).astype(np.uint8)


def printComparison(difference, original, reconstructedGrayLevels):
    print("max error:", difference.max())
    print("mean error:", difference.mean())
    print(
        "identical after rounding:", np.array_equal(original, reconstructedGrayLevels)
    )


def plotComparison(original, reconstructed, difference=None):
    panels = 2 if difference is None else 3

    plt.figure(figsize=(5 * panels, 4))
    plt.subplot(1, panels, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Imagem Original")
    plt.subplot(1, panels, 2)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("Imagem Reconstruída")

    if difference is not None:
        plt.subplot(1, panels, 3)
        plt.imshow(difference, cmap="gray", vmin=0, vmax=255)
        plt.title("Diferença")

    plt.show()


def rebuildSpectrum(magnitude, phase):
    return magnitude * np.exp(1j * phase)


def reconstructFromMagnitudeOnly(dft):
    magnitude = np.abs(dft)
    phase = np.zeros_like(magnitude)

    onlyMagnitudeSpectrum = rebuildSpectrum(magnitude, phase)

    return computeInverseFourierTransform(onlyMagnitudeSpectrum)


def reconstructFromPhaseOnly(dft):
    phase = np.angle(dft)
    magnitude = np.ones_like(phase)

    onlyPhaseSpectrum = rebuildSpectrum(magnitude, phase)

    return computeInverseFourierTransform(onlyPhaseSpectrum)


def plotPartialReconstructions(original, onlyMagnitude, onlyPhase):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Imagem Original")
    plt.subplot(1, 3, 2)
    plt.imshow(onlyMagnitude, cmap="gray")
    plt.title("Só Magnitude")
    plt.subplot(1, 3, 3)
    plt.imshow(onlyPhase, cmap="gray")
    plt.title("Só Fase")
    plt.show()


# ================================================================================================

# Parte: A
img = readImageFromMemory(
    imgName="image.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)
imgFloat = convertToFloat(img)

dftImg = computeFourierTransform(imgFloat)

reconstructedImg = computeInverseFourierTransform(dftImg)

plotComparison(img, reconstructedImg)

# Parte: B

img = readImageFromMemory(
    imgName="butterfly.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)
dftImg2 = computeFourierTransform(img)

onlyMagnitudeImg = reconstructFromMagnitudeOnly(dftImg2)
onlyPhaseImg = reconstructFromPhaseOnly(dftImg2)

plotPartialReconstructions(img, onlyMagnitudeImg, onlyPhaseImg)
