import numpy as np

x = np.array([1, 2, 3, 4])


def normalizeArray(array):
    return array / np.sum(array)


result = normalizeArray(x)

print(result)
