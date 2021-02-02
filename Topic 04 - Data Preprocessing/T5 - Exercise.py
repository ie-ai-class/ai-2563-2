from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from SBS import SBS
from sklearn.ensemble import RandomForestClassifier

#
from sklearn.datasets import load_breast_cancer


dataObj = load_breast_cancer()

X = dataObj.data
y = dataObj.target
cols = dataObj.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardization
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# dimReduceMethod = "L1"
# dimReduceMethod = "SBS"
dimReduceMethod = "randomForest"

n_neighbors = 3
reducedTo = 5

# Fit without dimensionality reduction
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_std, y_train)
print("---------------------")
print("Use all columns")
print("Training accuracy:", knn.score(X_train_std, y_train))
print("Test accuracy:", knn.score(X_test_std, y_test))
print("---------------------")
print(f"Dimensional reduction method: {dimReduceMethod}")


if dimReduceMethod == "L1":
    lr = LogisticRegression(solver="liblinear", penalty="l1", C=0.1, multi_class="ovr")
    lr.fit(X_train_std, y_train)
    print("Training accuracy:", lr.score(X_train_std, y_train))
    print("Test accuracy:", lr.score(X_test_std, y_test))
    coefs = lr.coef_.reshape(X.shape[1])
    idxs = np.argsort(np.abs(coefs))[::-1]
    for count, idx in enumerate(idxs[:10]):
        coef = coefs[idx]
        col = cols[idx]
        print(f"({count+1:2d}), Coefs: {coef:6.3f}, Name: {col.title():20s}")
    idxs = idxs[:reducedTo]

elif dimReduceMethod == "SBS":
    knn = KNeighborsClassifier(n_neighbors=2)
    # selecting features
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)
    # Print result
    for setf, score in zip(sbs.subsets_, sbs.scores_):
        print(f"#Features = {len(setf):3d}, Accuracy = {score:3.2f}")
    idxs = sbs.subsets_[-reducedTo]


elif dimReduceMethod == "randomForest":
    # Create objects
    forest = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    # Training
    forest.fit(X_train, y_train)
    # Extract importance measure
    importances = forest.feature_importances_
    # Sort array from based on importances from large to small
    idxs = np.argsort(importances)[::-1]
    # Print results
    for count, idx in enumerate(idxs):
        importance = importances[idx]
        col = cols[idx]
        print(f"{count+1:2d}) {col:30s} \t{importance:5.3f}")
    idxs = idxs[:reducedTo]


# Use reduced columns
X_train_std_reduced = X_train_std[:, idxs]
X_test_std_reduced = X_test_std[:, idxs]

print("---------------------")
print(f"Use {reducedTo:2d} features")
for count, idx in enumerate(idxs):
    col = cols[idx]
    print(f"{count+1:2d}) {col:30s}")
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_std_reduced, y_train)
print("Training accuracy:", knn.score(X_train_std_reduced, y_train))
print("Test accuracy:", knn.score(X_test_std_reduced, y_test))
