import numpy as np

npArray = np.array([1, 2, 3, 4])

def getOddNumbersArray(array):
  return array[array % 2 != 0].astype(int)

def getArrayToThePowerOf(array, powerOf):
  return array ** powerOf

oddArray = getOddNumbersArray(npArray)

result = getArrayToThePowerOf(oddArray, 2)

print(result)