import numpy as np

x = np.array([-3, -2, -1, 0, 1, 2, 3])


def changeArrayValuesLessBy(array, oldValue, newValue):
    array[array < oldValue] = newValue
    return array


result = changeArrayValuesLessBy(x, 0, 0)
print(result)
