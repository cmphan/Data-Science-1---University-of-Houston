#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Cuong Phan
# Decision Tree Example 
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics


# In[2]:


data = pd.read_csv('Buy_Computers.csv')
data.head()


# In[3]:


data.info()


# In[4]:


#identify target as class 
data['buys_computer'], class_names = pd.factorize(data['buys_computer'])


# In[5]:


print(class_names)


# In[6]:


print(data['buys_computer'])


# In[7]:


#Identify the predictors variables and encode any string variables 
#into equivalent integer code
data['age'],_ = pd.factorize(data['age'])
data['income'],_ = pd.factorize(data['income'])
data['student'],_ = pd.factorize(data['student'])
data['credit_rating'],_ = pd.factorize(data['credit_rating'])


# In[8]:


data.head()


# In[9]:


#type has changed from object to int64
data.info()


# In[10]:


#Select the predictor feature and target variable
X = data.iloc[:,:-1]
y = data.iloc[:,-1]


# In[11]:


#Split data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,
                                                    random_state=0)


# In[12]:


#Training/model fitting
#train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,
                                    random_state=0)


# In[13]:


dtree.fit(X_train, y_train)


# In[14]:


#Model parameters study
#use the model to make predictions with the test data
y_pred = dtree.predict(X_test)


# In[15]:


#how did the model perform?
count_misclassified = (y_test != y_pred).sum()


# In[16]:


print('Misclassified samples: {}'.format(count_misclassified))


# In[17]:


#Caculate and print accuracy
accuracy = metrics.accuracy_score(y_test,y_pred)
print('Accuracy ',accuracy)


# In[18]:


# Visualize data
#with open("buys_computers_classifier.txt", "w") as f:
    #tree.export_graphviz(dtree, out_file=f)


# In[19]:


import graphviz
feature_names = X.columns
dot_data = tree.export_graphviz(dtree, out_file=None, filled=True,
                                rounded=True,
                               feature_names=feature_names, 
                                class_names=class_names)


# In[20]:


graph = graphviz.Source(dot_data)


# In[21]:


graph


# In[ ]:




