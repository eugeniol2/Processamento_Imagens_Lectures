import numpy as np


def getEvenNumbersArray(array):
    return array[array % 2 == 0].astype(int)


# ================================================================================================

x = np.array([1, 2, 3, 4, 5, 6, 7])

result = getEvenNumbersArray(x).size
print(result)
