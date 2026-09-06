import numpy as np


x = np.array([1, 2, 3, 4, 5, 6, 7])

def getEvenNumbersArray(array):
  return array[array % 2 == 0].astype(int)

result = getEvenNumbersArray(x).size
print(result)