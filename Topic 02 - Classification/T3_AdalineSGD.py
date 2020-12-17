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

plt.close("all")

# =============================================================================
# Functions
# =============================================================================
def plot_decision_surface(y, X, W):
    # y is a 1D NumPy array of length n
    # X is a 2D NumPy array of shape (m+1,n). This has a column of 1's.
    # W is a 1D NumPy array of length m+1. The first element is the bias.
    # Note that this function requres a function yHat_(X,W).
    resolution = 0.02
    markers = ("s", "x", "o", "^", "v")
    # plot the decision surface
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x2_min, x2_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    Xg = np.array([xx1.ravel(), xx2.ravel()]).T
    xg0 = np.ones((Xg.shape[0], 1))
    Xg = np.hstack((xg0, Xg))
    Z = yHat_(Xg, W)
    Z = Z.reshape(xx1.shape)

    # plot area
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap="Set3")
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        px = X[y == cl, 1]
        py = X[y == cl, 2]
        plt.scatter(
            px,
            py,
            alpha=0.8,
            cmap="Pastel1",
            edgecolor="black",
            marker=markers[idx],
            label=cl,
        )

    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("./perceptron_2.png", dpi=300)
    plt.show()


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


def shuffleArray(y, X):
    n = y.shape[0]
    od = np.random.permutation(n)
    y = y[od]
    X = X[od, :]
    return (y, X)


# =============================================================================
# Program start
# =============================================================================
# Set parameter
param = "ex1"
paramSet = {
    "ex1": {"eta": 0.1, "tf": 3, "isStd": True},
}

eta = paramSet[param]["eta"]
tf = paramSet[param]["tf"]
isStd = paramSet[param]["isStd"]

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

# Standardization
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

# Initialize weight and bias
W = np.zeros(3)

for t in range(tf):

    # Shuffle training data
    (y, X) = shuffleArray(y, X)

    for i in range(y.shape[0]):

        # Extract Extract X^i, y^i
        yi = y[i]
        Xi = X[i, :]

        # Calculate phi(z^i)
        phi = phi_(Xi, W)
        diff_y_phi = yi - phi

        # Calculate deltaW_j
        deltaW = eta * diff_y_phi * Xi

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
plot_decision_surface(y, X, W)
