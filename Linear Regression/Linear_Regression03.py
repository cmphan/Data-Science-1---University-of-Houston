#Cuong Phan
#Linear Regression slide 4,5 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#generate random data_set
np.random.seed(0)
x=np.random.rand(100,1)
y=2 + 3*x + np.random.rand(100,1)
#sckit-learn implementation
#Model initialization
regression_model = LinearRegression()
#Fit the data(train the model)
regression_model.fit(x,y)
#Predict
y_predicted = regression_model.predict(x)
#model evaluation
rmse = mean_squared_error(x,y_predicted)
r2 = r2_score(y,y_predicted)
#printing values 
print('Slope:', regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error:',rmse)
print('R2 score:',r2)
#plotting values # data points
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
#predicted values
plt.plot(x,y_predicted,color='r')
plt.show()

#mean squared error
mse = np.sum((y_predicted-y)**2)
#root mean squared error
#m is the number of training examples
m = x.size
rmse = np.sqrt(mse/m)

#sum of square of residuals
ssr = np.sum((y_predicted-y)**2)

#total sum of squares
sst = np.sum((y-np.mean(y))**2)

#R2 score
r2_score = 1-(ssr/sst)