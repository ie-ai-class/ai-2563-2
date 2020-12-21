import numpy as np
import matplotlib.pyplot as plt


def plot_decision_surface(y, X, W, ML_type, filename=""):
    # y is a 1D NumPy array of length n
    # X is a 2D NumPy array of shape (m+1,n). This has a column of 1's.
    # W is a 1D NumPy array of length m+1. The first element is the bias.

    if (type(y) != np.ndarray) | (type(X) != np.ndarray) | (type(W) != np.ndarray):
        print("\n\nPlotting Error:The input array needs to be NumPy arrays.")
        return
    if len(X.shape) != 2:
        print("\n\nPlotting Error: X needs to be a 2D NumPy array.")
    if X.shape[1] != 3:
        print("\n\nPlotting Error: X needs to have 3 columns.")
        return

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

    if ML_type == "perceptron":
        z_ = np.dot(Xg, W)
        phi_ = np.where(z_ >= 0, 1, -1)
        yHat_ = phi_
    elif ML_type == "adaline":
        z_ = np.dot(Xg, W)
        phi_ = z_
        yHat_ = np.where(phi_ >= 0, 1, -1)
    elif ML_type == "logistic":
        z_ = np.dot(Xg, W)
        phi_ = 1.0 / (1.0 + np.exp(-np.clip(z_, -250, 250)))
        yHat_ = np.where(phi_ >= 0.5, 1, 0)

    Z = yHat_
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
    if filename:
        plt.savefig("./" + filename, dpi=300)
    plt.show()


def plot_decision_surface_general(y, X, W, yHat_):
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