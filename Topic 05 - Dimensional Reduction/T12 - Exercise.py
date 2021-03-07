import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# define dataset
""" n_features = 40
X, y = make_classification(
    n_samples=1000,
    n_features=n_features,
    n_informative=15,
    n_redundant=5,
    random_state=7,
    n_classes=10,
)
dfx = pd.DataFrame(X)
cols = [f"X{i}" for i in range(1, n_features + 1)]
dfx.columns = cols
dfx.to_csv("data_x.csv", index=False)
sry = pd.Series(y)
sry.to_csv("data_y.csv", index=False) """

# Read CSV files
dfx = pd.read_csv("data_x.csv")
sry = pd.read_csv("data_y.csv")

# Extract data
X = dfx.values
y = sry.values.ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scaling data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# No dimensional reduction
lr = LogisticRegression()
lr = lr.fit(X_train_std, y_train)
print(f"Use all {X.shape[1]} columns")
print("Training accuracy:", lr.score(X_train_std, y_train))
print("Test accuracy:", lr.score(X_test_std, y_test))
print("--------------------------------")

print("Use PCA")
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
exVar = np.cumsum(pca.explained_variance_ratio_)
n_components = 19
print(f"Choose n = {n_components}")
print(f"Cumulative explained variance = {exVar[n_components]}")
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
print("Training accuracy:", lr.score(X_train_pca, y_train))
print("Test accuracy:", lr.score(X_test_pca, y_test))
print("--------------------------------")

print("Use LDA")
lda = LDA()
X_train_lda = lda.fit_transform(X_train_std, y_train)
exVar = np.cumsum(lda.explained_variance_ratio_)
n_components = 4
print(f"Choose n = {n_components}")
print(f"Cumulative explained variance = {exVar[n_components]}")
lda = LDA(n_components=n_components)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
print("Training accuracy:", lr.score(X_train_lda, y_train))
print("Test accuracy:", lr.score(X_test_lda, y_test))
