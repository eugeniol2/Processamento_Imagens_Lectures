import numpy as np

x = np. array ([1 , 2, 3, 4, 5, 6, 7, 8, 9, 10])
n = 4

def getMovingAverage1D(array, nWindowSize):
  return np.convolve(array, np.ones(nWindowSize), mode='valid') / nWindowSize

result = getMovingAverage1D(x, n)

print(result)