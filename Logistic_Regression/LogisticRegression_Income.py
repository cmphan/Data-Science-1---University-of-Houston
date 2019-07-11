#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Cuong Phan
#Group 6
#Logistic Regression Classifier on Income Data


# In[1]:


#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#allow plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
#loading the dataset
data = pd.read_csv("adult.data",names=[
        "Age", "Workclass", "Sector", "Education", "Education-Num", "Marital-Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital-Gain", "Capital-Loss",
        "Hours per week", "Native-Country", "y"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
data.info()


# In[58]:


data.head()


# In[59]:


data.info()


# There are 15 attributes in the dataset
# 1. Age: a person's age, continuous 
# 2. Workclass: Types of work, categorical attribute: Private, Self-emp-not-inc, Self-emp-inc Federal-gov,Local-gov, Sate-gov, Without-pay, Never-Worked
# 3. Sector: 
# 4. Education: highest education a person has, categorical attribute: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
# 5. Education-Num: Number of education years, continious attribute 
# 6. Marital-Status: Categorical attribute: Married-civ-spouse, Divorced, Never-married, Seperated, Widowed, Married-spouse-absent, Married-AF-spouse
# 7. Occupation: Categorical attribute: Tech-support, Craft-repair, Other-servic, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
# 8. Relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
# 9. Race: Categorical attribute: Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
# 10. Sex: Categorical attribute: Female, Male
# 11. Captial-Gain: continuous
# 12. Capital-Loss: continuous
# 13. Hours-per-week: continuous attribute: number of hours a person work
# 14. Native-Country: categorical attribute: A person's citizenship 
# 15. The target variable: categorical attribute: whether the income is larger than 50,000 or not
# 

# In[61]:


#Check if there is any missing data
data.isnull().sum()


# In[90]:


#Calculate correlation and plot it
sns.heatmap(data.corr(), square=True)
plt.show()


# There are a strong correlation between Education and Education-Num so take a look at those columns:

# In[62]:


#Take a look at two columns:
data[['Education','Education-Num']].head(10)


# Education and Education Num represent the same information so we will only include Education-Num 
# in our final dataset as it has the order attribute.

# In[63]:


data.describe()


# Notice that Capital Gain and Capital Loss have min and IQR values are 0. So need to take a look at those values

# In[64]:


data['Capital-Gain'].value_counts()


# The majority of Capital Gain consists of 0. Check Capital Loss columns

# In[66]:


data['Capital-Loss'].value_counts()


# Capital-Loss even have more 0 values. So we need to drop those two columns

# In[2]:


data.drop("Capital-Gain", axis=1, inplace=True,)
data.drop("Capital-Loss", axis=1, inplace=True,)


# In[3]:


data.tail()


# We also notice from our dataset that there are many categorical attributes so we encode them into numerical. 
# 
# 

# In[4]:


#Create Dummy Variables to encode categorical variables into numerical variables
data_workclass = pd.get_dummies(data['Workclass'])
data_marital_status = pd.get_dummies(data['Marital-Status'])
data_occupation = pd.get_dummies(data['Occupation'])
data_relationship = pd.get_dummies(data['Relationship'])
data_race = pd.get_dummies(data['Race'])
data_gender = pd.get_dummies(data['Sex'])
data_country = pd.get_dummies(data['Native-Country'])


# In[5]:


#Create a data Frame from Encoded Variables
data_encoded = pd.concat([data_workclass,data_marital_status,data_occupation,data_relationship,data_race,data_gender,data_country],axis=1)


# In[6]:


#Extract numerical data from the imported dataset
data_numeric = data[['Age','Sector','Education-Num','Hours per week']]


# In[7]:


data_numeric.head()


# In[8]:


#Combine Enconded Data and Numerical Data into a Final Dataset
data_final = pd.concat([data_encoded,data_numeric],axis=1)


# In[9]:


#Convert y variable into a binary outcome (do they make over 50k or not)
def get_y(y):
    if y.find("<=")>-1:
        return 0
    else:
        return 1


# Now the data is processed, we can train a model and evaluate its performance

# In[10]:


#X are predictors and y is target variable
X = data_final
y = data['y'].apply(lambda y: get_y(y))


# In[11]:


#Apply Standard Scaler because attribute values are not 
#in the same range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[['Age','Sector','Education-Num','Hours per week']] = sc.fit_transform(X[['Age','Sector','Education-Num','Hours per week']])


# In[12]:



#Split data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)
#Build a Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[13]:


#Predict the values with test set
y_pred = model.predict(X_test)
print("Score value: ", model.score(X_train,y_train))


# The high value of score indicates that this Logislic Regression is a good model

# In[14]:


#Compute a confusion matrix to evaluate model performance
from sklearn.metrics import confusion_matrix
df_confusion = pd.crosstab(y_test, y_pred)
df_confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)
df_conf_norm = df_confusion/df_confusion.sum(axis=1)
print(df_conf_norm)
import matplotlib as mpl
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) #imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks,df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)  
plot_confusion_matrix(df_confusion)


# In[16]:


#Calculate confusion matrix precision 
confusion_matrix(y_test, y_pred)
def precision_recall(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)    
    tp = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    prec = tp / (tp+fp)
    rec = tp / (tp+fn)    
    return prec, rec
precision, recall = precision_recall(y_test, y_pred)
print('Precision: %f Recall %f' % (precision, recall))


# The precision is 0.918591. We can evaluate that the Logistic Regression model is good for prediction. 

# In[108]:


#AUC and ROC curve to evaluate model performance
from sklearn.metrics import roc_curve, roc_auc_score
##Computing false and true positive rates
fpr, tpr,_=roc_curve(y_test,y_pred,drop_intermediate=False)
score = roc_auc_score(y_test,y_pred)
print("Are under the ROC curve: ", score)
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# The true positive rate evaluation results also indicate a good model

# In[113]:


plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
coefs = pd.Series(model.coef_[0], index=X_train.columns)
coefs.sort_values()
coefs.plot(kind="bar")
plt.show()


# In[ ]:




