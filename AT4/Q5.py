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


def buildGaussianLowPassMask(imageNdArray, sigma):
    rows, cols = imageNdArray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.float64)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = np.exp(-(distance**2) / (2 * sigma**2))

    return mask


def applyMask(fshift, mask):
    return fshift * mask


def computeInverseFourierTransform(fshiftFiltered):
    fIshift = uncenterSpectrum(fshiftFiltered)
    imgBack = np.fft.ifft2(fIshift)
    imgBack = np.abs(imgBack)

    return imgBack


def plotIdealVsGaussian(idealSmall, gaussianSmall, idealLarge, gaussianLarge, small, large):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(idealSmall, cmap="gray")
    plt.title(f"Ideal (r = {small})")
    plt.subplot(2, 2, 2)
    plt.imshow(gaussianSmall, cmap="gray")
    plt.title(f"Gaussiano (σ = {small})")

    plt.subplot(2, 2, 3)
    plt.imshow(idealLarge, cmap="gray")
    plt.title(f"Ideal (r = {large})")
    plt.subplot(2, 2, 4)
    plt.imshow(gaussianLarge, cmap="gray")
    plt.title(f"Gaussiano (σ = {large})")

    plt.tight_layout()
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

gaussianMaskSigma10 = buildGaussianLowPassMask(imgFloat, sigma=10)
gaussianMaskSigma60 = buildGaussianLowPassMask(imgFloat, sigma=60)

gaussianSpectrumSigma10 = applyMask(centeredDftImg, gaussianMaskSigma10)
gaussianSpectrumSigma60 = applyMask(centeredDftImg, gaussianMaskSigma60)

gaussianImgSigma10 = computeInverseFourierTransform(gaussianSpectrumSigma10)
gaussianImgSigma60 = computeInverseFourierTransform(gaussianSpectrumSigma60)

idealMaskR10 = buildLowPassMask(imgFloat, r=10)
idealMaskR60 = buildLowPassMask(imgFloat, r=60)

idealSpectrumR10 = applyMask(centeredDftImg, idealMaskR10)
idealSpectrumR60 = applyMask(centeredDftImg, idealMaskR60)

idealImgR10 = computeInverseFourierTransform(idealSpectrumR10)
idealImgR60 = computeInverseFourierTransform(idealSpectrumR60)

plotIdealVsGaussian(
    idealImgR10, gaussianImgSigma10, idealImgR60, gaussianImgSigma60, small=10, large=60
)

