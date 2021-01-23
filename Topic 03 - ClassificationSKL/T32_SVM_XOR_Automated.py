#%%
import __main__ as main

try:
    hasattr(main, "__file__")
    from IPython import get_ipython

    get_ipython().magic("reset -sf")
    get_ipython().magic("clear")
except:
    pass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Create data
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
# Plot data
fig1, ax1 = plt.subplots(1, figsize=(5, 5))
ax1.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], marker="s", label="-1")
ax1.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], marker="x", label="1")
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.legend()
fig1.savefig("XOR.png", dpi=300)


param = "ex5"
paramSet = {
    "ex1": {"gamma": 0.01, "C": 10},
    "ex2": {"gamma": 0.1, "C": 10},
    "ex3": {"gamma": 1, "C": 10},
    "ex4": {"gamma": 10, "C": 10},
    "ex5": {"gamma": 0.1, "C": 0.01},
    "ex6": {"gamma": 0.1, "C": 1},
    "ex7": {"gamma": 0.1, "C": 100},
    "ex8": {"gamma": 0.1, "C": 1000},
}

for ex, param in paramSet.items():
    # Training
    svm = SVC(kernel="rbf", random_state=1, gamma=param["gamma"], C=param["C"],)
    svm.fit(X_xor, y_xor)

    # Plotting result
    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
    Z = svm.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    X = X_xor
    y = y_xor
    markers = ("s", "x")
    linestyles = (":", "--")
    fig2, ax2 = plt.subplots(1, figsize=(5, 5))
    ax2.contourf(xx1, xx2, Z, alpha=0.4, cmap="Set3")
    for idx, cl in enumerate(np.unique(y)):
        ax2.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.6,
            edgecolor="black",
            cmap="Set3",
            marker=markers[idx],
            label=cl,
        )
    ax2.set_title("γ = " + str(param["gamma"]) + ", C = " + str(param["C"]))
    ax2.set_xlim([xx1.min(), xx1.max()])
    ax2.set_ylim([xx2.min(), xx2.max()])
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    ax2.legend()
    fig2.savefig("T32_XOR_fit_" + ex + ".png", dpi=300)
