#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import sklearn


# In[2]:


# generate data
from sklearn.datasets.samples_generator import make_blobs
data = make_blobs(n_samples=100, n_features=2, centers=3,cluster_std=2.5)[0]


# In[3]:


# compute centers
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(data)
[center_1, center_2, center_3] = km.cluster_centers_


# In[4]:


# compute distance from each data point to closest center:
distance = [(min(distances),i) for i,distances in enumerate(km.transform(data))]


# In[9]:


distance


# In[5]:


# pick the top 10 outliers
indices_of_outliers = [row[1] for row in sorted(distance, key=lambda row: row[0], reverse=True)[:10]]


# In[8]:


figure(1,figsize=(10,6))
plot([row[0] for row in data],[row[1] for row in data],'b.')
plot(center_1[0],center_1[1], 'g*',ms=15)
plot(center_2[0],center_2[1], 'g*',ms=15)
plot(center_3[0],center_3[1], 'g*',ms=15)
# mark outliers by drawing a circle on top of the already-present point:
# plot([row[0] for row in data],[row[1] for row in data],'ro')
for i in indices_of_outliers:
    [x,y] = data[i]
    plot(x,y,'ro')


# In[ ]:




