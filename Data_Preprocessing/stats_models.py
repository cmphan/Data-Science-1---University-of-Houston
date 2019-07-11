#Cuong Phan
#Stats Models
#imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import datasets ##imports dataset from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library
#define the data/predictors as the pre-set feature names
df = pd.DataFrame(data.data, columns = data.feature_names)
#Put the target(housing value--MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
#without a constant
import statsmodels.api as sm
x = df["RM"]
y = target["MEDV"]

#Note the different in argument order
model = sm.OLS(y,x).fit()
predictions = model.predict(x) #make the predictions by the model

#Print out the statistics
model.summary()


