from pathlib import Path
from typing import Literal

import cv2
import numpy as np

ImageType = Literal["png", "bmp", "tiff", "webp", "pgm", "jpg"]
Axis = Literal[0, 1]


def readImageFromMemory(imgName, imgDirectory, readMode):
    fileBytes = np.fromfile(
        Path(__file__).parent / imgDirectory / imgName, dtype=np.uint8
    )
    imageNdArray = cv2.imdecode(fileBytes, readMode)

    return imageNdArray


def saveImage(imageNdArray, name, imgType: ImageType = "png"):
    OUTPUT_DIR = Path(__file__).parent / (Path(__file__).stem + "_output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    extension = "." + imgType.lstrip(".")
    path = OUTPUT_DIR / (name + extension)

    ok, encodedBytes = cv2.imencode(extension, imageNdArray)

    if not ok:
        raise ValueError(f"could not encode image: {path}")

    encodedBytes.tofile(path)

    return path


def convertImageToNegative(city):
    cityNegative = 255 - city
    return cityNegative


def revertImageOnGivenAxis(imageNdArray, axis: Axis):
    return np.flip(m=imageNdArray, axis=axis)


def convertImageToRange(imageNdArray, newMin, newMax):
    currentMin = imageNdArray.min()
    currentMax = imageNdArray.max()

    normalized = (imageNdArray - currentMin) / (currentMax - currentMin)
    rescaled = normalized * (newMax - newMin) + newMin

    return rescaled


def invertEvenRows(imageNdArray):
    inverted = imageNdArray.copy()
    inverted[::2] = imageNdArray[::2, ::-1]

    return inverted


def mirrorTopHalfOntoBottom(imageNdArray):
    height = imageNdArray.shape[0]
    middle = height // 2

    mirrored = imageNdArray.copy()
    mirrored[middle:] = imageNdArray[:middle][::-1]

    return mirrored


def normValues(array):
    return array / 255


def applyGammaCorrection(imageNdArray, gamma):
    normalized = normValues(imageNdArray)
    corrected = normalized ** (1.0 / gamma)

    return np.round(corrected * 255).astype(np.uint8)


def binarizeByT(imageNdArray, T):
    return np.where(imageNdArray > T, 255, 0).astype(np.uint8)


# ================================================================================================


originalImg = readImageFromMemory(
    imgName="baboon_monocromatica.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

baboonBinary = binarizeByT(imageNdArray=originalImg, T=128)

saveImage(imageNdArray=baboonBinary, name="baboon_binary", imgType="png")
