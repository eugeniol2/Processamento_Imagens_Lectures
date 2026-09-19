import numpy as np


def convertToNumber(array):
    return array.astype(int)


def getArrayMinusLastIndex(array):
    return convertToNumber(array[:-1])


def getArrayMinusFirstIndex(array):
    return convertToNumber(array[1:])


def getQ1Results(array):
    subtractionResults = getArrayMinusLastIndex(array) - getArrayMinusFirstIndex(array)
    return np.sum(subtractionResults < 0)


# ================================================================================================

array = np.array([False, True, False, True])

Result = getQ1Results(array)

print(Result)
