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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import inspect
from PlotFunction2 import plot_decision_surface2

plt.close("all")
# =============================================================================
# Program start
# =============================================================================
#%% Model parameters
eta = 0.1

# Read data
iris = datasets.load_iris()

# Check methods and data
# print(inspect.getmembers(iris))
# print(dir(iris))
# print(iris.DESCR)
# print(iris.feature_names)
# print(iris.data)
# print(iris.target_names)
# print(iris.target)

# Extract the last 2 columns
X = iris.data[:, 2:4]
y = iris.target

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Verify stratification
# print(np.bincount(y))
# print(np.bincount(y_train))
# print(np.bincount(y_test))

# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create object (estimator)
ppn = Perceptron(eta0=eta, random_state=1, verbose=1, n_iter_no_change=10)

# Training
ppn.fit(X_train_std, y_train)

# Prediction
y_pred = ppn.predict(X_test_std)

# Misclassification from the test samples
sumMiss = (y_test != y_pred).sum()

# Accuracy score from the test samples
accuracyScore = accuracy_score(y_test, y_pred)

print(f"Misclassified examples: {sumMiss}")
print(f"Accuracy score: {accuracyScore}")

# Plot decision regions
plot_decision_surface2(
    X_train_std, X_test_std, y_train, y_test, ppn, filename="output.png"
)
