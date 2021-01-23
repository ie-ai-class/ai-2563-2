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
from sklearn.tree import DecisionTreeClassifier
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

param = "ex3"
paramSet = {
    "ex1": {"criterion": "gini", "max_depth": 4},
    "ex2": {"criterion": "entropy", "max_depth": 4},
    "ex3": {"criterion": "gini", "max_depth": 8},
}
params = paramSet[param]

# Creating objects
tree_model = DecisionTreeClassifier(
    criterion=params["criterion"], max_depth=params["max_depth"], random_state=1
)

# Fiting
tree_model.fit(X_train_std, y_train)

# Predict test samples
y_pred = tree_model.predict(X_test_std)

# Misclassification from the test samples
sumMiss = (y_test != y_pred).sum()

# Accuracy score from the test samples
accuracyScore = accuracy_score(y_test, y_pred)

print(f"Misclassified examples: {sumMiss}")
print(f"Accuracy score: {accuracyScore}")

# Plot dicision surface
filenamePNG = "T34_tree_boundary_" + param + ".png"
plot_decision_surface2(
    X_train_std, X_test_std, y_train, y_test, tree_model, filename=filenamePNG
)

# Visualization 1: Plot tree
""" fig, ax = plt.subplots(1, figsize=(5, 5))
tree.plot_tree(
    tree_model,
    feature_names=iris.feature_names[2:4],
    class_names=iris.target_names,
    filled=True,
)
filenamePDF = "T34_tree_visualize1_" + param + ".pdf"
fig.savefig(filenamePDF) """


# Visualization 2: graphviz
import graphviz

# DOT data
""" dot_data = tree.export_graphviz(
    tree_model,
    out_file=None,
    feature_names=iris.feature_names[2:4],
    class_names=iris.target_names,
    filled=True,
)

# Draw graph
graph = graphviz.Source(dot_data, format="png")
graph """

# Visualization 3: dtreeviz
""" import graphviz
from dtreeviz.trees import dtreeviz  # remember to load the package
import cairosvg

viz = dtreeviz(
    tree_model,
    X_train_std,
    y,
    target_name="target",
    feature_names=iris.feature_names[2:4],
    class_names=list(iris.target_names),
)

viz
filenameSVG = "T34_tree_visualize3_" + param + ".svg"
viz.save(filenameSVG)

filenamePDF2 = "T34_tree_visualize3_" + param + ".pdf"
cairosvg.svg2pdf(url=filenameSVG, write_to=filenamePDF2) """

