import numpy as np


def getEuclideanDistanceBetweenToPoints(a, b):
    return np.linalg.norm(a - b)


# ================================================================================================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = getEuclideanDistanceBetweenToPoints(a, b)

print(result)
