import numpy as np

a = np.array ([1, 2, 3])
b = np.array ([4, 5, 6])

def getEuclideanDistanceBetweenToPoints(a, b):
  return np.linalg.norm(a-b)


result = getEuclideanDistanceBetweenToPoints(a, b)

print(result)