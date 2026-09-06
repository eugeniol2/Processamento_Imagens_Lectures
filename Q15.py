import numpy as np
from typing import Literal

rng = np.random.default_rng(42)

A = rng.integers(low=1, high=3, size=(10, 10))


def getAggregateByAxis(
    array,
    axis: Literal["horizontal", "vertical"] = "horizontal",
    op: Literal["sum", "mean", "median"] = "sum",
):

    axisMap = {
        "horizontal": 1,
        "vertical": 0,
    }

    operations = {"sum": np.sum, "mean": np.mean, "median": np.median}

    axisDirection = axisMap[axis]
    numpyFunction = operations[op]

    return numpyFunction(array, axis=axisDirection)


lineSum = getAggregateByAxis(A, "horizontal")
lineMean = getAggregateByAxis(A, "horizontal", "mean")
columnSum = getAggregateByAxis(A, "vertical")
columnMean = getAggregateByAxis(A, "vertical", "mean")



print("A", lineSum)
print("B", columnSum)
print("C", lineMean)
print("D", columnMean)
