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


def computeMagnitude(fshift):
    return np.abs(fshift)


def computePhaseSpectrum(fshift):
    return np.angle(fshift)


def rebuildSpectrum(magnitude, phase):
    return magnitude * np.exp(1j * phase)


def computeInverseFourierTransform(fshiftFiltered):
    fIshift = uncenterSpectrum(fshiftFiltered)
    imgBack = np.fft.ifft2(fIshift)
    imgBack = np.abs(imgBack)

    return imgBack


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


def plotPhaseMagnitudeExperiment(
    imgA,
    imgB,
    magnitudeAPhaseB,
    magnitudeBPhaseA,
    bottomTitles=("(i) magnitude de A + fase de B", "(ii) magnitude de B + fase de A"),
):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(imgA, cmap="gray")
    plt.title("A (original)")
    plt.subplot(2, 2, 2)
    plt.imshow(imgB, cmap="gray")
    plt.title("B (original)")

    plt.subplot(2, 2, 3)
    plt.imshow(magnitudeAPhaseB, cmap="gray")
    plt.title(bottomTitles[0])
    plt.subplot(2, 2, 4)
    plt.imshow(magnitudeBPhaseA, cmap="gray")
    plt.title(bottomTitles[1])

    plt.tight_layout()
    plt.show()


# ================================================================================================

imgA = readImageFromMemory(
    imgName="image.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)
imgB = readImageFromMemory(
    imgName="butterfly.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

imgAFloat = convertToFloat(imgA)
imgBFloat = convertToFloat(imgB)

dftA = computeFourierTransform(imgAFloat)
dftB = computeFourierTransform(imgBFloat)

centeredDftA = centerSpectrum(dftA)
centeredDftB = centerSpectrum(dftB)

magnitudeA = computeMagnitude(centeredDftA)
phaseA = computePhaseSpectrum(centeredDftA)

magnitudeB = computeMagnitude(centeredDftB)
phaseB = computePhaseSpectrum(centeredDftB)

magnitudeAPhaseBSpectrum = rebuildSpectrum(magnitudeA, phaseB)
magnitudeBPhaseASpectrum = rebuildSpectrum(magnitudeB, phaseA)

magnitudeAPhaseBImg = computeInverseFourierTransform(magnitudeAPhaseBSpectrum)
magnitudeBPhaseAImg = computeInverseFourierTransform(magnitudeBPhaseASpectrum)

plotPhaseMagnitudeExperiment(
    imgA, imgB, magnitudeAPhaseBImg, magnitudeBPhaseAImg
)



dftAOnlyMagnitude = reconstructFromMagnitudeOnly(dftA)
dftBOnlyMagnitude = reconstructFromMagnitudeOnly(dftB)


plotPhaseMagnitudeExperiment(
    imgA,
    imgB,
    centerSpectrum(dftAOnlyMagnitude),
    centerSpectrum(dftBOnlyMagnitude),
    bottomTitles=("Só magnitude de A (fase = 0)", "Só magnitude de B (fase = 0)"),
)


dftAOnlyPhase = reconstructFromPhaseOnly(dftA)
dftBOnlyPhase = reconstructFromPhaseOnly(dftB)

plotPhaseMagnitudeExperiment(
    imgA,
    imgB,
    dftAOnlyPhase,
    dftBOnlyPhase,
    bottomTitles=("Só fase de A (magnitude = 1)", "Só fase de B (magnitude = 1)"),
)

