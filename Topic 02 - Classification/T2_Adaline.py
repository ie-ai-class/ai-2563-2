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
    return z


def yHat_(X, W):
    phi = phi_(X, W)
    return np.where(phi >= 0, 1, -1)


def numFalse_(y, X, W):
    yh = yHat_(X, W)
    return (yh != y).sum()


def J_(y, X, W):
    phi = phi_(X, W)
    diff_y_phi = y - phi
    J = 0.5 * np.sum(diff_y_phi ** 2)
    return J


# =============================================================================
# Program start
# =============================================================================
# Set parameter
param = "ex3"
paramSet = {
    "ex1": {"eta": 0.0001, "tf": 30, "isStd": False},
    "ex2": {"eta": 0.01, "tf": 10, "isStd": False},
    "ex3": {"eta": 0.01, "tf": 10, "isStd": True},
}

eta = paramSet[param]["eta"]
tf = paramSet[param]["tf"]
isStd = paramSet[param]["isStd"]

# Read data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)

# Extract y values and perform feature extraction
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

# Extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# Append a columne of X0
x0 = np.ones((X.shape[0], 1))
X = np.hstack((x0, X))

# Standardization
if isStd:
    # Keep the original data
    X_ori = X.copy()

    # Transform the data
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

# Initialize weight and bias
W = np.zeros(3)

for t in range(tf):

    # Calculate phi(z^i)
    phi = phi_(X, W)

    # Calculate deltaW_j
    diff_y_phi = y - phi
    deltaW0 = eta * (diff_y_phi * X[:, 0]).sum()
    deltaW1 = eta * (diff_y_phi * X[:, 1]).sum()
    deltaW2 = eta * (diff_y_phi * X[:, 2]).sum()
    deltaW = np.array([deltaW0, deltaW1, deltaW2])

    # Update W
    W = W + deltaW

    # Count misclassification
    numFalse = numFalse_(y, X, W)

    # Calculate cost
    J = J_(y, X, W)

    # Print result
    print(
        f"Epoch = {t+1:2d},  " f"numFalse = {numFalse:3d},  " f"J = {J:5.2f},  " "W =",
        np.array2string(W, formatter={"float_kind": lambda x: "%.4f" % x}),
    )

# Plot
plot_decision_surface(y, X, W, ML_type="adaline")
