import numpy as np


def changeArrayValuesLessBy(array, oldValue, newValue):
    array[array < oldValue] = newValue
    return array


def get1DArraySum(array):
    return np.sum(array)


v = np.array([1, -2, 3, 4, -5, 6, -7, 8, -9])

onlyPositiveArray = changeArrayValuesLessBy(v, 0, 0)

result = get1DArraySum(onlyPositiveArray)
print(result)
