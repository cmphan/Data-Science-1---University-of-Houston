#Cuong Phan
#K-Folds Cross Validation
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#dataset
X = np.array([[1,2],[3,4],[1,2],[3,4]]) #create an array
y = np.array([1,2,3,4]) #create another array
#Import K-Folds
from sklearn.model_selection import KFold 
kf = KFold(n_splits=2) #Define the split into 2 folds
kf.get_n_splits(X) #returns the number of splitting iterations in the cross-validator
print(kf)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
#Validation Part
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
#Perform 6 fold cross validation
scores = cross_val_score(model, df, y, cv=6)
print("Cross-validated scores:", scores)

#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y,predictions)
accuracy = metrics.r2_score(y,predictions)
print("Cross-Predicted Accuracy:" , accuracy)
