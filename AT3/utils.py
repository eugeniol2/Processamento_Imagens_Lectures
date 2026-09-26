import inspect
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
    # __file__ here would be utils.py, so the folder is named after the caller
    callerFile = Path(inspect.currentframe().f_back.f_code.co_filename)

    OUTPUT_DIR = callerFile.parent / (callerFile.stem + "_output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    extension = "." + imgType.lstrip(".")
    path = OUTPUT_DIR / (name + extension)

    ok, encodedBytes = cv2.imencode(extension, imageNdArray)

    if not ok:
        raise ValueError(f"could not encode image: {path}")

    encodedBytes.tofile(path)

    return path
