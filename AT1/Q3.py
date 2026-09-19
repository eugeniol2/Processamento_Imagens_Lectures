import numpy as np


def getDivisibleBy(array, value):
    x = array[array % value == 0].astype(int)
    return x


def get1DArraySum(array):
    return np.sum(array)


# ================================================================================================

npArray = np.array([1, 5, 12, 15, 20, 22])

arrayDivisiblebyFive = getDivisibleBy(npArray, 5)

result = get1DArraySum(arrayDivisiblebyFive)

print(result)
