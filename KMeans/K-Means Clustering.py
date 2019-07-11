#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[83]:


#Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"


# In[84]:


train = pd.read_csv(train_url)


# In[85]:


test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"


# In[86]:


test = pd.read_csv(test_url)


# In[87]:


print("****Train_Set***")
train.head()


# In[88]:


print("****Test_Set***")
test.head()


# In[89]:


print("****Train_Set***")
train.describe()


# In[90]:


print("****Test_Set***")
test.describe()


# In[91]:


print(train.columns.values)


# In[92]:


#missing values for the train set
train.isna().head()


# In[93]:


#missing values for the test set
test.isna().head()


# In[94]:


#total number of missing values
print("****In the Train Set***")
print(train.isna().sum())


# In[95]:


print("****In the Test Set***")
print(test.isna().sum())


# In[96]:


#Fill missing values with mean colum values in the train set
train.fillna(train.mean(),inplace=True)


# In[97]:


#Fill missing values with mean colum values in the test set
test.fillna(train.mean(),inplace=True)


# In[98]:


print("****In the Train Set***")
print(train.isna().sum())


# In[99]:


print("****In the Test Set***")
print(test.isna().sum())


# In[100]:


#Categorical: Survived, Sex, and Embarked. Ordinal: Pclass
#Continuous: Age, Fare, Discrete, SibSp, Parch
#Ticket and Cabin. Ticket is the mix of numeric and alphanumeric data types
#Cabin is alphanumeric


# In[101]:


train['Ticket'].head()


# In[102]:


train['Cabin'].head()


# In[103]:


#Survival count with respect to Pclass:
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[104]:


#Survival count with respect to Sex:
train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[105]:


#Survival count with respect to SibSp:
train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[106]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist,'Age',bins=20)


# In[107]:


grid = sns.FacetGrid(train,col='Survived',row='Pclass',size=2.2, aspect=1.6)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()


# In[108]:


#build a K - Mean
train.info()


# In[121]:


#feature enginerring i.e features like Name, Ticket, Cabin and
#Embarked do not have any impact on the survival staus of the passengers
train = train.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
test = test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)


# In[122]:


#The only none numeric value is sex let's convert it


# In[123]:


labelEncoder = LabelEncoder()


# In[124]:


labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])


# In[125]:


train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])


# In[126]:


# Investigate non-numeric data left
train.info()


# In[127]:


# Label survived should be dropped, test set does not have one anyway
test.info()


# In[128]:


y = np.array(train['Survived'])


# In[129]:


#drop Survival Colum from train
train=train.drop(['Survived'],axis=1)


# In[131]:


X = np.array(train)


# In[132]:


kmeans = KMeans(n_clusters=2) #Want cluster passenger records in 2:
#Survived or Not survived
kmeans.fit(X)


# In[135]:


#How many are predicted correctly based on y
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
print(correct/len(X))


# In[138]:


#model was able to cluster correctly with 50% accuracy
#tweak some parameters of the model itself to enhance model accuracy such as
#algorithm, max-ter, n-jobs
kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm='auto', n_jobs=1) #Want cluster passenger records in 2:

kmeans.fit(X)


# In[139]:


#How many are predicted correctly based on y
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
print(correct/len(X))


# In[140]:


# a slight change because we have not scaled the values 
#when feeding the model
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)


# In[142]:


#+ 12 after scaling
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
print(correct/len(X))


# In[ ]:




