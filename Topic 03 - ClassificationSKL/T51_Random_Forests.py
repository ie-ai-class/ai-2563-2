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
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

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
    "ex1": {
        "criterion": "gini",
        "n_estimators": 25,
        "max_samples": None,
        "max_features": "auto",
        "max_depth": None,
    },
    "ex2": {
        "criterion": "gini",
        "n_estimators": 100,
        "max_samples": None,
        "max_features": "auto",
        "max_depth": 2,
    },
    "ex3": {
        "criterion": "gini",
        "n_estimators": 200,
        "max_samples": 20,
        "max_features": "auto",
        "max_depth": 2,
    },
}
params = paramSet[param]

# Creating objects
forest = RandomForestClassifier(
    criterion=params["criterion"],
    n_estimators=params["n_estimators"],
    max_samples=params["max_samples"],
    max_features=params["max_features"],
    max_depth=params["max_depth"],
    random_state=1,
    n_jobs=2,
)

# Fiting
forest.fit(X_train_std, y_train)

# Predict test samples
y_pred = forest.predict(X_test_std)

# Misclassification from the test samples
sumMiss = (y_test != y_pred).sum()

# Accuracy score from the test samples
accuracyScore = accuracy_score(y_test, y_pred)

print(f"Misclassified examples: {sumMiss}")
print(f"Accuracy score: {accuracyScore}")


# Plot dicision surface
filenamePNG = "T51_forest_decision_" + param + ".png"
plot_decision_surface2(
    X_train_std, X_test_std, y_train, y_test, forest, filename=filenamePNG
)

nTrees = 5
fig, ax = plt.subplots(1, nTrees, figsize=(40, 8))
for i in range(0, nTrees):
    plt.sca(ax[i])
    tree.plot_tree(
        forest.estimators_[i],
        feature_names=iris.feature_names[2:4],
        class_names=iris.target_names,
        filled=True,
    )
filenamePDF = "T51_forrest_decision_tree_" + param + ".pdf"
fig.savefig(filenamePDF)

from pdf2image import convert_from_path

filenameIMG = "T51_forrest_decision_tree_" + param + ".png"
pages = convert_from_path(filenamePDF, 500)
for page in pages:
    page.save(filenameIMG, "PNG")
