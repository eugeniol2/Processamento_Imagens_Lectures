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

highPassMaskR10 = buildHighPassMask(imgFloat, r=10)
highPassMaskR60 = buildHighPassMask(imgFloat, r=60)

highPassSpectrumR10 = applyMask(centeredDftImg, highPassMaskR10)
highPassSpectrumR60 = applyMask(centeredDftImg, highPassMaskR60)

highPassImgR10 = computeInverseFourierTransform(highPassSpectrumR10)
highPassImgR60 = computeInverseFourierTransform(highPassSpectrumR60)

plotComparison(img, highPassImgR10)
plotComparison(img, highPassImgR60)
