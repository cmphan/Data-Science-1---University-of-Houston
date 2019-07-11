#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
iris = datasets.load_iris()
digits = datasets.load_digits()


# In[3]:


#The digits dataset
digits = datasets.load_digits()


# In[6]:


#images attribute of the dataset 8x8:
images_and_labels = list(zip(digits.images, digits.target))
for index, (image,label) in enumerate(images_and_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training: %i' %label)


# In[7]:


#flatten the image, to turn data in a(samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))


# In[8]:


classifier = svm.SVC(gamma=0.001)


# In[12]:


#We learn the digits on the first half of the digits
classifier.fit(data[:n_samples//2], digits.target[:n_samples//2])


# In[13]:


#Now predict the value of the digit on the second half:
expected = digits.target[n_samples//2:]
predicted = classifier.predict(data[n_samples//2:])


# In[15]:


print("Classification report for classifier %s:\n%s\n"
    %(classifier,metrics.classification_report(expected,predicted)))


# In[17]:


print("Confusion matrix: \n%s"
    % metrics.confusion_matrix(expected,predicted))


# In[21]:


images_and_predictions = list(zip(digits.images[n_samples//2:], predicted))
for index, (image,prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2,4,index+5)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction: %i' %prediction)
plt.show()


# In[ ]:




