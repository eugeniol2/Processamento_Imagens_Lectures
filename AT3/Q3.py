import cv2
import numpy as np

from utils import readImageFromMemory, saveImage

GAUSSIAN_3X3 = (
    np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ]
    )
    / 16
)

GAUSSIAN_5X5 = (
    np.array(
        [
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1],
        ]
    )
    / 273
)

GAUSSIAN_7X7 = (
    np.array(
        [
            [0, 0, 1, 2, 1, 0, 0],
            [0, 3, 13, 22, 13, 3, 0],
            [1, 13, 59, 97, 59, 13, 1],
            [2, 22, 97, 159, 97, 22, 2],
            [1, 13, 59, 97, 59, 13, 1],
            [0, 3, 13, 22, 13, 3, 0],
            [0, 0, 1, 2, 1, 0, 0],
        ]
    )
    / 1003
)


def boxKernel(size):
    return np.ones((size, size)) / (size * size)


def applyKernel(imageNdArray, kernel):
    kernelHeight, kernelWidth = kernel.shape

   
    windows = np.lib.stride_tricks.sliding_window_view(
        imageNdArray, (kernelHeight, kernelWidth)
    )


    insideEachWindow = (-2, -1)
    filtered = (windows * kernel).sum(axis=insideEachWindow)

    return np.round(filtered).astype(np.uint8)


# ================================================================================================

aerialImg = readImageFromMemory(
    imgName="aerial_view.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

aerialBox3 = applyKernel(aerialImg, boxKernel(3))
aerialBox5 = applyKernel(aerialImg, boxKernel(5))
aerialBox7 = applyKernel(aerialImg, boxKernel(7))

aerialGaussian3 = applyKernel(aerialImg, GAUSSIAN_3X3)
aerialGaussian5 = applyKernel(aerialImg, GAUSSIAN_5X5)
aerialGaussian7 = applyKernel(aerialImg, GAUSSIAN_7X7)

saveImage(imageNdArray=aerialBox3, name="aerial_box_3x3", imgType="png")
saveImage(imageNdArray=aerialBox5, name="aerial_box_5x5", imgType="png")
saveImage(imageNdArray=aerialBox7, name="aerial_box_7x7", imgType="png")

saveImage(imageNdArray=aerialGaussian3, name="aerial_gaussian_3x3", imgType="png")
saveImage(imageNdArray=aerialGaussian5, name="aerial_gaussian_5x5", imgType="png")
saveImage(imageNdArray=aerialGaussian7, name="aerial_gaussian_7x7", imgType="png")
