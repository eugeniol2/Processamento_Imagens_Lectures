import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


def splitArrayOnBlocksByN(array, n):
  amountOfBlocks = array.size // n
  newarr = array.reshape(n, amountOfBlocks)
  return newarr

def getMultiple1DArraySum(array):
  return np.sum(array, axis=1)

splittedArrays = splitArrayOnBlocksByN(x, 3)

result = getMultiple1DArraySum(splittedArrays)


print(result)