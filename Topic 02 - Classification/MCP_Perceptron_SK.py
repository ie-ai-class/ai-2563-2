from IPython import get_ipython
get_ipython().magic('reset -sf') 
get_ipython().magic('clear') 


from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

# =============================================================================
# Functions
# =============================================================================
def plot_decision_surface_SK(y,X,ppn):
    resolution=0.02
    markers = ('s', 'x', 'o', '^', 'v')
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    
    Xg = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = ppn.predict(Xg)
    Z = Z.reshape(xx1.shape)
    
    # plot area
    plt.contourf(xx1, xx2, Z, alpha=0.4,cmap='Set3')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        px = X[y == cl, 0]
        py = X[y == cl, 1]
        plt.scatter(px, py, 
                    alpha=0.8, cmap='Pastel1',
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
    
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('./perceptron_2.png', dpi=300)
    plt.show()

# =============================================================================
# Program start
# =============================================================================
# Model parameters
eta=0.1

# Read data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)

# Extract y values and perform feature extraction
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Extract sepal length and petal length
X = df.iloc[0:100,[0,2]].values

# =============================================================================
# SciKitLearn Part 
# =============================================================================
# Standardization
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

#Create object (estimator)
ppn = Perceptron(eta0=eta, random_state=1,verbose=1)

#Training
ppn.fit(X_std,y)

#Extracting coefficients
W = np.append(ppn.intercept_.flatten() , ppn.coef_.flatten())
print('\n\nW =', np.array2string(W,formatter={'float_kind':lambda x: "%.4f" % x}))

#Plotting decision surface
plot_decision_surface_SK(y,X_std,ppn)