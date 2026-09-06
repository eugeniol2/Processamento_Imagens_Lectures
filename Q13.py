import numpy as np

x1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])


def subtractTwoArrays(array1, array2):
  return array1 - array2


def isPalindrome(array):
  mergedArray = subtractTwoArrays(array, np.flip(array))
  return np.all(mergedArray == 0)

result = isPalindrome(x1)

print(result)
