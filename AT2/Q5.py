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


def splitIntoBlocks(imageNdArray, gridRows, gridCols):
    height, width = imageNdArray.shape
    blockHeight = height // gridRows
    blockWidth = width // gridCols

    grid = imageNdArray.reshape(gridRows, blockHeight, gridCols, blockWidth)

    return grid.transpose(0, 2, 1, 3).reshape(
        gridRows * gridCols, blockHeight, blockWidth
    )


def reorderBlocks(blocks, blockOrder):
    return blocks[blockOrder.ravel() - 1]


def joinBlocks(blocks, gridRows, gridCols):
    blockHeight, blockWidth = blocks.shape[1:]

    grid = blocks.reshape(gridRows, gridCols, blockHeight, blockWidth)

    return grid.transpose(0, 2, 1, 3).reshape(
        gridRows * blockHeight, gridCols * blockWidth
    )


# ================================================================================================

MOSAIC = np.array(
    [
        [6, 11, 13, 3],
        [8, 16, 1, 9],
        [12, 14, 2, 7],
        [4, 15, 10, 5],
    ]
)

originalImg = readImageFromMemory(
    imgName="baboon_monocromatica.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

baboonBlocks = splitIntoBlocks(originalImg, gridRows=4, gridCols=4)
baboonReordered = reorderBlocks(baboonBlocks, blockOrder=MOSAIC)
baboonMosaic = joinBlocks(baboonReordered, gridRows=4, gridCols=4)

saveImage(imageNdArray=baboonMosaic, name="baboon_mosaic", imgType="png")



