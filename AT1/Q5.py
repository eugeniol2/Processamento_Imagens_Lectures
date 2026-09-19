import numpy as np


def changeArrayValuesLessBy(array, oldValue, newValue):
    array[array < oldValue] = newValue
    return array


# ================================================================================================

x = np.array([-3, -2, -1, 0, 1, 2, 3])

result = changeArrayValuesLessBy(x, 0, 0)
print(result)
