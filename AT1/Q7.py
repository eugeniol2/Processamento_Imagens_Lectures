import numpy as np


def getMovingAverage1D(array, nWindowSize):
    return np.convolve(array, np.ones(nWindowSize), mode="valid") / nWindowSize


# ================================================================================================

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n = 4

result = getMovingAverage1D(x, n)

print(result)
