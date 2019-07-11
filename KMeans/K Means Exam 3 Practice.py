#!/usr/bin/env python
# coding: utf-8

# In[66]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# In[30]:


#Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"


# In[31]:


train = pd.read_csv(train_url)


# In[32]:


train.head()


# In[33]:


test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"


# In[34]:


test = pd.read_csv(test_url)


# In[35]:


test


# In[36]:


#Missing values in train
train.isna().sum()


# In[38]:


#Missing values in test
test.isna().sum()


# In[39]:


#Fill missing values with mean colum values in the train set
train.fillna(train.mean(),inplace=True)
test.fillna(train.mean(),inplace=True)


# In[40]:


train = train.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
test = test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)


# In[41]:


train


# In[43]:


labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])


# In[44]:


y_train = np.array(train['Survived'])


# In[45]:


train=train.drop(['Survived'],axis=1)


# In[46]:


X_train = np.array(train)


# In[60]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)


# In[61]:


def kmeans(X, n_clusters):
    ss = StandardScaler()
    X = ss.fit_transform(X)
    km = KMeans(n_clusters=n_clusters)
    km.fit(X)
    y_pred = km.predict(X)
    return km
#The best number of clusters is 3
km = kmeans(X_train,2)


# In[62]:


#How many are predicted correctly based on y
def correction(X, y, kmeans):
    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1,len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct +=1
    print(correct/len(X))
correction(X_train, y_train, km)


# In[63]:


from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering


# In[64]:


algorithms = []
algorithms.append(KMeans(n_clusters=2, random_state=1))
algorithms.append(AffinityPropagation())
algorithms.append(SpectralClustering(n_clusters=2, random_state=1,
                                     affinity='nearest_neighbors'))
algorithms.append(AgglomerativeClustering(n_clusters=2))


# In[67]:


data = []
for algo in algorithms:
    algo.fit(X_train)
    data.append(({
        'ARI': metrics.adjusted_rand_score(y_train, algo.labels_),
        'AMI': metrics.adjusted_mutual_info_score(y_train, algo.labels_),
        'Homogenity': metrics.homogeneity_score(y_train, algo.labels_),
        'Completeness': metrics.completeness_score(y_train, algo.labels_),
        'V-measure': metrics.v_measure_score(y_train, algo.labels_),
        'Silhouette': metrics.silhouette_score(X_train, algo.labels_)}))

results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity',
                                           'Completeness', 'V-measure', 
                                           'Silhouette'],
                       index=['K-means', 'Affinity', 
                              'Spectral', 'Agglomerative'])

results


# In[71]:


def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
        distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
        return distances


# In[72]:


centroids = km.cluster_centers_
print("centroids:", centroids)


# In[73]:


from scipy.spatial import distance


# In[ ]:


distances_0 = []
distance = distance.euclidean(centroids, )

