import numpy as np
from typing import Literal

rng = np.random.default_rng(42)

A = rng.integers(low=0, high=100, size=(4, 5, 3))


def getMaxValueFlatIndexFromArray(array):
    return np.argmax(array)


def getCoordsFromFlatIndex(index, array):
    return np.unravel_index(index, array.shape)


def getAggregateByAxis3D(
    array, axis: int = 0, op: Literal["sum", "mean", "median", "max"] = "sum"
):
    operations = {"sum": np.sum, "mean": np.mean, "median": np.median, "max": np.max}
    numpyFunction = operations[op]

    return numpyFunction(array, axis=axis)


firstDimentionSum = getAggregateByAxis3D(A, op="sum", axis=0)

secondDimentionSum = getAggregateByAxis3D(A, op="sum", axis=1)

thirdDimentionSum = getAggregateByAxis3D(A, op="sum", axis=2)

firstDimentionMean = getAggregateByAxis3D(A, op="mean", axis=0)

secondDimentionMean = getAggregateByAxis3D(A, op="mean", axis=1)

thirdDimentionMean = getAggregateByAxis3D(A, op="mean", axis=2)

firstDimentionHighest = getAggregateByAxis3D(A, op="max", axis=0)

secondDimentionHighest = getAggregateByAxis3D(A, op="max", axis=1)

thirdDimentionHighest = getAggregateByAxis3D(A, op="max", axis=2)


print(
    "A axis= 0 \n", firstDimentionSum
)  # percorre sobre os elementos  [i][j][k] na primeira posição marcada em 'i'
print(
    "A axis= 1 \n", secondDimentionSum
)  # percorre sobre os elementos [i][j][k] na segunda posição marcada em 'j'
print(
    "A axis= 2 \n", thirdDimentionSum
)  # percorre sobre os elementos  [i][j][k] na terceira posição marcada em 'k'


print(
    "B axis= 0 \n", firstDimentionMean
)  # percorre sobre os elementos  [i][j][k] na primeira posição marcada em 'i'
print(
    "B axis= 1 \n", secondDimentionMean
)  # percorre sobre os elementos [i][j][k] na segunda posição marcada em 'j'
print(
    "B axis= 2 \n", thirdDimentionMean
)  # percorre sobre os elementos  [i][j][k] na terceira posição marcada em 'k'


print(
    "C axis= 0 \n", firstDimentionHighest
)  # percorre sobre os elementos  [i][j][k] na primeira posição marcada em 'i'
print(
    "C axis= 1 \n", secondDimentionHighest
)  # percorre sobre os elementos [i][j][k] na segunda posição marcada em 'j'
print(
    "C axis= 2 \n", thirdDimentionHighest
)  # percorre sobre os elementos  [i][j][k] na terceira posição marcada em 'k'


flatIndex = getMaxValueFlatIndexFromArray(A)

coords = getCoordsFromFlatIndex(flatIndex, A)

print("i, j, k", coords)
