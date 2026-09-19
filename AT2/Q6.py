from pathlib import Path
from typing import Literal

import cv2
import numpy as np

ImageType = Literal["png", "bmp", "tiff", "webp", "pgm", "jpg"]


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


def mergeImages(imageA, imageB, weightA, weightB):
    merged = imageA * weightA + imageB * weightB

    return np.round(merged).astype(np.uint8)


# ================================================================================================


baboonImg = readImageFromMemory(
    imgName="baboon_monocromatica.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

butterflyImg = readImageFromMemory(
    imgName="butterfly.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

merged01 = mergeImages(imageA=baboonImg, imageB=butterflyImg, weightA=0.2, weightB=0.8)
merged02 = mergeImages(imageA=baboonImg, imageB=butterflyImg, weightA=0.5, weightB=0.5)
merged03 = mergeImages(imageA=baboonImg, imageB=butterflyImg, weightA=0.8, weightB=0.2)

saveImage(imageNdArray=merged01, name="merged_1", imgType="png")
saveImage(imageNdArray=merged02, name="merged_2", imgType="png")
saveImage(imageNdArray=merged03, name="merged_3", imgType="png")

