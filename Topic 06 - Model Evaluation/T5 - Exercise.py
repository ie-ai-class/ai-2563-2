import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Load data
dataObj = load_digits()
X = dataObj.data
y = dataObj.target


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=1
)


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = Pipeline(
    [
        ("scl", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("clf", SVC(random_state=1)),
    ]
)
