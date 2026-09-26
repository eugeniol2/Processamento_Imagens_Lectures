import cv2
import numpy as np

from utils import readImageFromMemory, saveImage


def buildGaussianKernel(size, sigma):
    # the x and y values: distances from the center
    #   size 21 -> [-10, ..., 0, ..., 10]
    coordinates = np.arange(size) - size // 2

    # x² + y² for every cell (x as a column, y as a row, crossed by broadcasting)
    squaredDistance = coordinates[:, None] ** 2 + coordinates[None, :] ** 2

    # G(x, y) = e^(-(x² + y²) / (2 · σ²))
    kernel = np.exp(-squaredDistance / (2 * sigma**2))

    # G(x, y) / Σ G: the weights add up to 1, so the blur keeps the brightness
    return kernel / kernel.sum()


def applyKernel(imageNdArray, kernel):
    kernelHeight, kernelWidth = kernel.shape

    windows = np.lib.stride_tricks.sliding_window_view(
        imageNdArray, (kernelHeight, kernelWidth)
    )
  
  # [ Position 0,0 ]  ──>  Holds an inner 3x3 matrix  ──>  Axes (-2, -1)
  # [ Position 0,1 ]  ──>  Holds an inner 3x3 matrix  ──>  Axes (-2, -1)
  # [ Position 0,2 ]  ──>  Holds an inner 3x3 matrix  ──>  Axes (-2, -1)

    insideEachWindow = (-2, -1)
    filtered = (windows * kernel).sum(axis=insideEachWindow)

    return np.round(filtered).astype(np.uint8) # convert to unsigned 8-bit integers


def paddingByRepeatingBorder(imageNdArray, kernelSize):
    margin = kernelSize // 2

    return np.pad(imageNdArray, margin, mode="mean") # pad the image by repeating the border pixels


def applyGaussianBlur(imageNdArray, kernelSize, sigma):
    return applyKernel(imageNdArray, buildGaussianKernel(kernelSize, sigma)) # extract the kernel and apply it to the image


def divideByBlurredVersion(imageNdArray, blurred):
    return imageNdArray / np.maximum(blurred, 1) # prevent division by zero


def convertRatioToGrayLevels(ratio):
    sketch = np.clip(ratio, 0, 1) * 255 # scale to 0-255 range

    return np.round(sketch).astype(np.uint8) # convert to unsigned 8-bit integers


# ================================================================================================

watchImg = readImageFromMemory(
    imgName="gray_watch.png",
    imgDirectory="imagens de entrada",
    readMode=cv2.IMREAD_GRAYSCALE,
)

watchPadded = paddingByRepeatingBorder(watchImg, kernelSize=21)
watchBlurred = applyGaussianBlur(watchPadded, kernelSize=21, sigma=4)
watchRatio = divideByBlurredVersion(watchImg, watchBlurred)
watchSketch = convertRatioToGrayLevels(watchRatio)

saveImage(imageNdArray=watchSketch, name="gray_watch_sketch", imgType="png")
