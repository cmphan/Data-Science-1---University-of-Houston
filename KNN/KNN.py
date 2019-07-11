#!/usr/bin/env python
# coding: utf-8

# In[6]:


X = [[0],[1],[2],[3]]
y = [0,0,1,1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
print(neigh.predict([[1.1]]))


# In[8]:


print(neigh.predict_proba([[0.9]]))


# In[14]:


def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[X]:
            correct +=1
    return(correct/float(len(testSet)))*100
        


# In[15]:


testSet = [[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
predictions = ['a','a','a']
accuracy = getAccuracy(testSet,predictions)
print(accuracy)


# In[ ]:




