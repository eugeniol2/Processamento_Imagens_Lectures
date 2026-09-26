import cv2
import numpy as np

from utils import readImageFromMemory, saveImage


def rotateClockwise90(imageNdArray):
    height, width = imageNdArray.shape
    rotated = np.empty((width, height), dtype=imageNdArray.dtype)

    for i in range(height):
        for j in range(width):
            rotated[j, height - i - 1] = imageNdArray[i, j]

    return rotated


def rotate180(imageNdArray):
    height, width = imageNdArray.shape
    rotated = np.empty((height, width), dtype=imageNdArray.dtype)

    for i in range(height):
        for j in range(width):
            rotated[height - i - 1, width - j - 1] = imageNdArray[i, j]

    return rotated


def rotateClockwise270(imageNdArray):
    height, width = imageNdArray.shape
    rotated = np.empty((width, height), dtype=imageNdArray.dtype)

    for i in range(height):
        for j in range(width):
            rotated[width - j - 1, i] = imageNdArray[i, j]

    return rotated


# ================================================================================================

baboonImg = readImageFromMemory(
    imgName="baboon_monocromatica.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

baboonRotated90 = rotateClockwise90(baboonImg)
baboonRotated180 = rotate180(baboonImg)
baboonRotated270 = rotateClockwise270(baboonImg)

saveImage(imageNdArray=baboonRotated90, name="baboon_rotated_90", imgType="png")
saveImage(imageNdArray=baboonRotated180, name="baboon_rotated_180", imgType="png")
saveImage(imageNdArray=baboonRotated270, name="baboon_rotated_270", imgType="png")

