import numpy as np


def normalizeArray(array):
    return array / np.sum(array)


# ================================================================================================

x = np.array([1, 2, 3, 4])

result = normalizeArray(x)

print(result)
