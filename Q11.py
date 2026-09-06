import numpy as np

v = np. array ([10, 7, 4, 3, 2])

def compareArrayItemsDiff(array, axis = 0):
  return np.diff(array, axis=axis)


result = compareArrayItemsDiff(v)

print(result)