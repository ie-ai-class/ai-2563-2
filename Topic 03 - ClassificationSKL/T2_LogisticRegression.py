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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import inspect
from PlotFunction2 import plot_decision_surface2

plt.close("all")
# =============================================================================
# Program start
# =============================================================================

# param = "ex2"
paramSet = {
    "ex1": {"solver": "lbfgs", "C": 0.0001, "max_iter": 100},
    "ex2": {"solver": "lbfgs", "C": 0.01, "max_iter": 100},
    "ex3": {"solver": "lbfgs", "C": 1, "max_iter": 100},
    "ex4": {"solver": "lbfgs", "C": 100, "max_iter": 100},
    "ex5": {"solver": "lbfgs", "C": 10000, "max_iter": 100},
}

# Read data
iris = datasets.load_iris()

# Extract the last 2 columns
X = iris.data[:, 2:4]
y = iris.target

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

for param, _ in paramSet.items():
    lr = LogisticRegression(
        random_state=1,
        verbose=0,
        solver=paramSet[param]["solver"],
        C=paramSet[param]["C"],
        max_iter=paramSet[param]["max_iter"],
    )

    # Training
    lr.fit(X_train_std, y_train)

    # Prediction
    y_pred = lr.predict(X_test_std)

    # Misclassification from the test samples
    sumMiss = (y_test != y_pred).sum()

    # Accuracy score from the test samples
    accuracyScore = accuracy_score(y_test, y_pred)

    print(f"Misclassified examples: {sumMiss}")
    print(f"Accuracy score: {accuracyScore}")
    print(f"Norm of W: {np.linalg.norm(lr.coef_)}")
    # Print the probability of each class
    # lr.predict_proba(X_test_std[:2,:])

    # Plot decision regions
    plot_decision_surface2(
        X_train_std, X_test_std, y_train, y_test, lr, filename=f"output_{param}.png"
    )
