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


def applyMask(fshift, mask):
    return fshift * mask


def computeInverseFourierTransform(fshiftFiltered):
    fIshift = uncenterSpectrum(fshiftFiltered)
    imgBack = np.fft.ifft2(fIshift)
    imgBack = np.abs(imgBack)

    return imgBack


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


# ================================================================================================

img = readImageFromMemory(
    imgName="image.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)
imgFloat = convertToFloat(img)

dftImg = computeFourierTransform(imgFloat)
centeredDftImg = centerSpectrum(dftImg)

lowPassMaskR10 = buildLowPassMask(imgFloat, r=10)
lowPassMaskR60 = buildLowPassMask(imgFloat, r=60)

filteredSpectrumR10 = applyMask(centeredDftImg, lowPassMaskR10)
filteredSpectrumR60 = applyMask(centeredDftImg, lowPassMaskR60)

filteredImgR10 = computeInverseFourierTransform(filteredSpectrumR10)
filteredImgR60 = computeInverseFourierTransform(filteredSpectrumR60)

plotComparison(img, filteredImgR10)
plotComparison(img, filteredImgR60)
