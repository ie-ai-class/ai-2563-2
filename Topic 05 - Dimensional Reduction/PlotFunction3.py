import numpy as np
import matplotlib.pyplot as plt


def plot_reduced_dim(X, y, plotType="PCA", filename=""):

    colors = ["r", "b", "g"]
    markers = ["s", "x", "o"]

    fig, ax = plt.subplots()

    for l, c, m in zip(np.unique(y), colors, markers):
        ax.scatter(X[y == l, 0], X[y == l, 1], c=c, label=l, marker=m)

    if plotType == "PCA":
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    elif plotType == "LDA":
        ax.set_xlabel("LDA 1")
        ax.set_ylabel("LDA 2")

    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    if filename:
        loc = filename.find(".")
        fname = filename[:loc]
        ext = filename[loc:]
        fig.savefig("./" + fname + "_2" + ext, dpi=300)