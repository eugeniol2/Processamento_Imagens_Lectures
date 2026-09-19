import numpy as np
from typing import Literal


def subtractTwoArrays(array1, array2):
    return array1 - array2


def getAggregateByAxis(
    array,
    axis=None,
    op: Literal["sum", "mean", "median", "max", "min"] = "sum",
    keepdims=False,
):
    operations = {
        "sum": np.sum,
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "min": np.min,
    }
    numpyFunction = operations[op]

    return numpyFunction(array, axis=axis, keepdims=keepdims)


def normalizeToRange01(array, axis=None):
    minimo = getAggregateByAxis(array, axis=axis, op="min", keepdims=True)
    maximo = getAggregateByAxis(array, axis=axis, op="max", keepdims=True)

    return subtractTwoArrays(array, minimo) / subtractTwoArrays(maximo, minimo)


# ================================================================================================

np.set_printoptions(precision=3, suppress=True)

rng = np.random.default_rng(5)

A = rng.uniform(low=-50, high=50, size=(10, 10))

matrixMin = getAggregateByAxis(A, op="min")
matrixMax = getAggregateByAxis(A, op="max")

normalized = normalizeToRange01(A)
normalizedByRow = normalizeToRange01(A, axis=1)

print("Matriz original A", A.shape, "\n", A)

print("Amin =", matrixMin)
print("Amax =", matrixMax)

print("\n Matriz normalizada", normalized.shape, "\n", normalized)

print("Desafio adicional")

print(normalizedByRow)

print("por linha, min de cada linha:", normalizedByRow.min(axis=1))
print("por linha, max de cada linha:", normalizedByRow.max(axis=1))
