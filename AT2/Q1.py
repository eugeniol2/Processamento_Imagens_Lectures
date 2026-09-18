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
    OUTPUT_DIR = Path(__file__).parent / "output"
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



city = readImageFromMemory(imgName="city.png", imgDirectory="imagens de entrada", readMode=cv2.IMREAD_GRAYSCALE)

city_negative_ndArray = convertImageToNegative(city)


saveImage(imageNdArray=city_negative_ndArray, name="city_negative", imgType="png")

