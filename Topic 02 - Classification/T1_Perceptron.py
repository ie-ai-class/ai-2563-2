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
    return np.where(z >= 0, 1, -1)


def yHat_(X, W):
    return phi_(X, W)


def numFalse_(y, X, W):
    yh = yHat_(X, W)
    return (yh != y).sum()


# =============================================================================
# Program start
# =============================================================================
# Set parameter
eta = 0.1
tf = 10

# Read/format data
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

# Initialize weight and bias
W = np.zeros(3)

for t in range(tf):

    for i in range(y.shape[0]):

        # Extract Extract X^i, y^i
        yi = y[i]
        Xi = X[i, :]

        # Calculate yHat^i
        yHat = yHat_(Xi, W)
        dy = yi - yHat

        # Calculate deltaW_j
        deltaW = eta * dy * Xi

        # Update W
        W = W + deltaW

    # Count misclassification
    numFalse = numFalse_(y, X, W)

    # Print result
    print(
        f"Epoch = {t+1:2d},  " f"numFalse = {numFalse:3d},  " "W =",
        np.array2string(W, formatter={"float_kind": lambda x: "%.4f" % x}),
    )

# Plotting decision surface
# plot_decision_surface(y, X, W, ML_type="perceptron", filename="output.png")
plot_decision_surface(y, X, W, ML_type="perceptron")
