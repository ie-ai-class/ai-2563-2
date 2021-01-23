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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PlotFunction2 import plot_decision_surface2

plt.close("all")
# =============================================================================
# Program start
# =============================================================================

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

param = "ex2"
paramSet = {
    "ex1": {"C": 1.0, "gamma": 0.2},
    "ex2": {"C": 1.0, "gamma": 100},
    "ex3": {"C": 10, "gamma": 0.1},
}
params = paramSet[param]

svm = SVC(kernel="rbf", gamma=params["gamma"], C=params["C"], random_state=1)
svm.fit(X_train_std, y_train)

# Predict test samples
y_pred = svm.predict(X_test_std)

# Misclassification from the test samples
sumMiss = (y_test != y_pred).sum()

# Accuracy score from the test samples
accuracyScore = accuracy_score(y_test, y_pred)

print(f"Misclassified examples: {sumMiss}")
print(f"Accuracy score: {accuracyScore}")

filenamePNG = "T33_SVM_" + param + ".png"
plot_decision_surface2(
    X_train_std, X_test_std, y_train, y_test, svm, filename=filenamePNG
)
