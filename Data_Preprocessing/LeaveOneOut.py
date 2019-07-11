#Cuong Phan
#Leave One Out
#Imports
import numpy as np
from sklearn.model_selection import LeaveOneOut
X = np.array([[1,2],[3,4]])
y = np.array([1,2])
loo = LeaveOneOut()
loo.get_n_splits(X)
for train_index, test_index in loo.split(X):
    print("train:", train_index, "validation: ", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]