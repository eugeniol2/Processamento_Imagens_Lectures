import numpy as np


def compareArrayItemsDiff(array, axis=0):
    return np.diff(array, axis=axis)


# ================================================================================================

v = np.array([10, 7, 4, 3, 2])

result = compareArrayItemsDiff(v)

print(result)
