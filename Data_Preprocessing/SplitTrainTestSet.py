#Cuong Phan
#Split Train Test Set
#import libraries
import numpy as np 
import pandas as pd #for dataframes
import matplotlib.pyplot as plt #for plotting graphs
#import dataset
data = pd.read_csv("HR_comma_sep.csv")
#Data-Preprocessing
#Label encoding of categorical using sklearn
#import Label Encoder 
from sklearn import preprocessing
#create LabelEncoder
le = preprocessing.LabelEncoder()
#Converting string labels into numers.
data['salary'] = le.fit_transform(data['salary'])
#Rename the column from "sales" to deparment 
data = data.rename(columns = {'sales' : 'department'})
#combine "technical", "support" and "IT" these three together and call them "technical"
data['department'].unique()
data['department'] = np.where(data['department'] == 'support','technical', data['department'])
data['department'] = np.where(data['department'] == 'IT','technical', data['department'])
#Converting string labels in numbers
data['salary'] = le.fit_transform(data['salary'])
data['department'] = le.fit_transform(data['department'])
data.head()
#Split into Train and Test set
#Spliting data into independent and explanatory values
X = data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company'
         ,'Work_accident', 'promotion_last_5years', 'department', 'salary']]
y = data['left']
#Import train_test_split function
from sklearn.model_selection import train_test_split
#Split data into training set and test set
# 70% and 30%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=42)

#print out X_train, X_test, y_train, y_test values
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
#fit a model 
from sklearn import datasets, linear_model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
#print the first 5 predictions
predictions[0:5]
##The line/model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
print("Score: ", model.score(X_test,y_test))