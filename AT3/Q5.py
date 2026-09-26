import cv2
import numpy as np

from utils import readImageFromMemory, saveImage

LAPLACIAN_KERNEL = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]
)


def paddingByRepeatingBorder(imageNdArray, kernelSize):
    margin = kernelSize // 2

    return np.pad(imageNdArray, margin, mode="edge") # pad the image by repeating the border pixels


def applyKernel(imageNdArray, kernel):
    kernelHeight, kernelWidth = kernel.shape

    windows = np.lib.stride_tricks.sliding_window_view(
        imageNdArray.astype(np.float64), (kernelHeight, kernelWidth)
    )

    insideEachWindow = (-2, -1)

    return (windows * kernel).sum(axis=insideEachWindow)


def convertToEdgeMagnitude(laplacian):
    return np.clip(np.abs(laplacian), 0, 255).astype(np.uint8) # convert to unsigned 8-bit integers


def rescaleToGrayLevels(laplacian):
    normalized = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min()) # normalize to 0-1 range

    return np.round(normalized * 255).astype(np.uint8) # convert to unsigned 8-bit integers


# ================================================================================================

peppersImg = readImageFromMemory(
    imgName="gray_peppers.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

peppersPadded = paddingByRepeatingBorder(peppersImg, kernelSize=3)
peppersLaplacian = applyKernel(peppersPadded, LAPLACIAN_KERNEL)
peppersEdges = convertToEdgeMagnitude(peppersLaplacian)
peppersRescaled = rescaleToGrayLevels(peppersLaplacian)

saveImage(imageNdArray=peppersEdges, name="gray_peppers_laplacian_edges", imgType="png")
