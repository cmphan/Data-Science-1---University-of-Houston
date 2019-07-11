#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#Function importing Dataset
def importdata():
    balance_data = pd.read_csv('balance-scale.csv', sep=',',header=None)
    #Printing dataset shape: 
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    #Print the dataset observations
    print("Dataset: ", balance_data.head())
    #implement data info, missing value check... 
    return balance_data 


# In[7]:


#Function to split the dataset 
def splitdataset(balance_data):
    #Seperate the target variable
    X = balance_data.values[:, 1: 5]
    y = balance_data.values[:, 0]
    #Spliting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=100)
    return X, y, X_train, X_test, y_train, y_test


# In[8]:


#Function to perform training data with giniIndex
def train_using_gini(X_train, X_test, y_train):
    #Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100,
                                     max_depth=3, min_samples_leaf=5)
    #Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# In[9]:


#Function to make predictions
def prediction(X_test, clf_object):
    #Prediction on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values: ")
    print(y_pred)
    return y_pred


# In[10]:


#Function to calculate accuracy 
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
    print("Accuracy: ",accuracy_score(y_test,y_pred)*100)
    print("Report: ",classification_report(y_test,y_pred))
          


# In[11]:


#Function to perform training with entropy
def train_using_entropy(X_train, X_test, y_train):
    #Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy",random_state=100,
                                     max_depth=3, min_samples_leaf=5)
    #Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# In[12]:


# Function to perform Naive Bayes Classifier
def train_using_naive(X_train, X_test, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
# Predicting the Test set results
    return classifier


# In[13]:


#Driver code
def main():
    #Building Phase
    data = importdata()
    data.info()
    X, y, X_train, X_test, y_train, y_test = splitdataset(data)
    #Train using gini
    clf_gini = train_using_gini(X_train,X_test,y_train)
    #Train using entropy
    clf_entropy = train_using_gini(X_train,X_test,y_train)
    #Operational Phase
    #Prediction with Gini
    y_pred_gini = prediction(X_test,clf_gini)
    print("Results Using Gini Index: ")
    cal_accuracy(y_test,y_pred_gini)
    df_confusion = pd.crosstab(y_test, y_pred_gini)
    plot_confusion_matrix(df_confusion)
    #Prediction with Entropy
    y_pred_entropy = prediction(X_test,clf_entropy)
    print("Results Using Gini Entropy: ")
    cal_accuracy(y_test,y_pred_gini)
    #Training with Naive Bayes Classifier
    clf_naive = train_using_naive(X_train, X_test, y_train)
    #Prediction with Naive Bayes
    y_pred_naive = prediction(X_test,clf_naive)
    print("Results Using Naive Bayes: ")
    cal_accuracy(y_test,y_pred_naive)


# In[14]:


#Calling main function
if __name__== "__main__":
    main()


# In[ ]:




