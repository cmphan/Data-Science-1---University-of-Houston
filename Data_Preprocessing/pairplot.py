#Cuong Phan
#Pair Plot
#imports
import pandas as pd
import seaborn as sns 
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np




#read data into a DataFrame
data = pd.read_csv("advertising.csv", index_col = 0)
data.head()

#shape of the DataFrame
data.shape

#visual the relationship between the features and the response using scatterplots 
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size = 7, aspect = 0.7)
sns.pairplot(data)
sns.pairplot(data.dropna())
sns.pairplot(data, diag_kind = "kde")
sns.pairplot(data, diag_kind = "kde", markers="+", plot_kws=dict(s=50,edgecolor="b", linewidth=1), diag_kws=dict(shade=True))
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars=['Sales'], size = 7, aspect = 0.7)
sns.pairplot(data, x_vars=['TV','Radio'], y_vars=['Newspaper','Sales'])