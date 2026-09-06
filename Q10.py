import numpy as np

v = np. array ([1, 2, 3, 4, 5])


def compareArrayItemsDiff(array, axis = 0):
  return np.diff(array, axis=axis)

def getIsArrayCrescent(array):
  differenceResultArray = compareArrayItemsDiff(array)
  return np.all(differenceResultArray >= 0)



print(getIsArrayCrescent(v))