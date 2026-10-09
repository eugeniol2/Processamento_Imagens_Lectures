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


def normalizePhase(phase):
    return (phase + np.pi) / (2 * np.pi)


def plotOriginalAndSpectra(original, magnitude, phaseNorm):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Imagem Original")
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude, cmap="gray")
    plt.title("Magnitude")
    plt.subplot(1, 3, 3)
    plt.imshow(phaseNorm, cmap="gray")
    plt.title("Fase")
    plt.show()


# ================================================================================================

img = readImageFromMemory(
    imgName="image.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)
img = convertToFloat(img)

dftImg = computeFourierTransform(img)

centeredDftImg = centerSpectrum(dftImg)

magnitudeSpectrum = computeMagnitudeSpectrum(centeredDftImg)
phaseSpectrum = computePhaseSpectrum(centeredDftImg)

plotOriginalAndSpectra(img, magnitudeSpectrum, normalizePhase(phaseSpectrum))
