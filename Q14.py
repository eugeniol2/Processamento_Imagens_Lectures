import numpy as np

rng = np.random.default_rng(42)

A = rng.integers(low=1, high=2, size=(10, 10))
B = rng.integers(low=1, high=2, size=(10, 10))


def sumTwoMatrices(mA, mB):
    return mA + mB


def getAbsoluteDiff(mA, mB):
    return np.abs(mA - mB)


def getMean(matrice):
    return np.mean(matrice)


def getBothMean(mA, mB):
    return getMean(mA), getMean(mB)


def newMatriceWithHighestValues(mA, mB):
    return np.maximum(mA, mB)


matricesSum = sumTwoMatrices(A, B)
matricesDiff = getAbsoluteDiff(A, B)
matricesMean = getBothMean(A, B)
maximumNewMatrice = newMatriceWithHighestValues(A, B)

print("A", matricesSum)
print("B", matricesDiff)
print("C", matricesMean)
print("D", maximumNewMatrice)
