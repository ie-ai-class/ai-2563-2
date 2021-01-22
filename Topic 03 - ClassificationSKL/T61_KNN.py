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
from sklearn.metrics import accuracy_score
from PlotFunction2 import plot_decision_surface2
from sklearn.neighbors import KNeighborsClassifier

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

param = "ex4"
paramSet = {
    "ex1": {"n_neighbors": 1, "p": 2, "algorithm": "auto",},
    "ex2": {"n_neighbors": 3, "p": 2, "algorithm": "auto",},
    "ex3": {"n_neighbors": 5, "p": 2, "algorithm": "auto",},
    "ex4": {"n_neighbors": 5, "p": 1, "algorithm": "auto",},
}
params = paramSet[param]

# Creating objects
knn = KNeighborsClassifier(
    n_neighbors=params["n_neighbors"],
    p=params["p"],
    metric="minkowski",
    algorithm=params["algorithm"],
)

# Fiting
knn.fit(X_train_std, y_train)

# Predict test samples
y_pred = knn.predict(X_test_std)

# Misclassification from the test samples
sumMiss = (y_test != y_pred).sum()

# Accuracy score from the test samples
accuracyScore = accuracy_score(y_test, y_pred)

print(f"Misclassified examples: {sumMiss}")
print(f"Accuracy score: {accuracyScore}")


# Plot dicision surface
filenamePNG = "T61_KNN_decision_" + param + ".png"
plot_decision_surface2(
    X_train_std, X_test_std, y_train, y_test, knn, filename=filenamePNG
)
