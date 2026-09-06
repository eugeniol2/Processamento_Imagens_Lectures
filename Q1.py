import numpy as np

array = np.array([False, True, False, True])

def convertToNumber(array):
  return array.astype(int)

def getArrayMinusLastIndex(array):
  return convertToNumber(array[:-1])

def getArrayMinusFirstIndex(array):
  return convertToNumber(array[1:])

def getQ1Results(array):
  subtractionResults = getArrayMinusLastIndex(array) - getArrayMinusFirstIndex(array)
  return np.sum(subtractionResults < 0)

Result = getQ1Results(array)

print(Result)