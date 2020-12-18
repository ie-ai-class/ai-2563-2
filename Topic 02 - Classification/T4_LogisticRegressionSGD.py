import __main__ as main

try:
    hasattr(main, "__file__")
    from IPython import get_ipython

    get_ipython().magic("reset -sf")
    get_ipython().magic("clear")
except:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PlotFunction import plot_decision_surface

plt.close("all")
# =============================================================================
# Functions
# =============================================================================
def z_(X, W):
    n = W.shape[0]
    X = X.reshape(-1, n)
    return np.dot(X, W)


def phi_(X, W):
    z = z_(X, W)
    phi = 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))  # Note the clip function
    return phi


def yHat_(X, W):
    phi = phi_(X, W)
    return np.where(phi >= 0.5, 1, 0)  # Note the cut-off value


def numFalse_(y, X, W):
    yh = yHat_(X, W)
    return (yh != y).sum()


def J_(y, X, W):
    phi = phi_(X, W)
    J = -np.sum(y * np.log(phi) + (1 - y) * np.log(1 - phi))
    return J


def shuffleArray(y, X):
    n = y.shape[0]
    od = np.random.permutation(n)
    y = y[od]
    X = X[od, :]
    return (y, X)


# =============================================================================
# Program start
# =============================================================================
# Model parameters
param = "ex1"
paramSet = {
    "ex1": {"eta": 0.1, "tf": 10, "isStd": True},
}

eta = paramSet[param]["eta"]
tf = paramSet[param]["tf"]
isStd = paramSet[param]["isStd"]

# Read data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)

# Extract y values and perform feature extraction
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)  # Note that y is either 0 or 1

# Extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# Append a columne of X0
x0 = np.ones((X.shape[0], 1))
X = np.hstack((x0, X))

# Standardization
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

# Initialize weight and bias
W = np.zeros(3)

for t in range(tf):
    for i in range(y.shape[0]):
        yi = y[i]
        Xi = X[i, :]
        phi = phi_(Xi, W)
        diff_y_phi = yi - phi

        deltaW0 = eta * diff_y_phi
        deltaW = eta * diff_y_phi * Xi[1:]

        W[0] = W[0] + deltaW0
        W[1:] = W[1:] + deltaW

    numFalse = numFalse_(y, X, W)
    J = J_(y, X, W)

    print(
        f"Epoch = {t+1:2d},  " f"numFalse = {numFalse:3d},  " f"J = {J:5.2f},  " "W =",
        np.array2string(W, formatter={"float_kind": lambda x: "%.4f" % x}),
    )

    (y, X) = shuffleArray(y, X)

plot_decision_surface(y, X, W, ML_type="logistic")
