#Cuong Phan
#Linear Regression slide 2
import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([5,20,14,32,22,38])
print(x)
print(y)
model = LinearRegression()
model.fit(x,y) #calculate the optimal values of the weight b0 and b1 using the existing input and output(x and y) as the arguments
r_sq = model.score(x,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope', model.coef_) #You can notice that .intercept_ is a scalar while .coef_ is an array.
#b0(intercept) = 5.63 (approximately) illustrates that your model predicts the reponse 5.63 when x is zero. The value b1(slope) =  0.54 means that
#the predicted response rises by 0.54 when x is increased by one.
new_model = LinearRegression().fit(x,y.reshape((-1,1))) #two-dimensional array as well
print('intercept:', new_model.intercept_)
print('slope', new_model.coef_)
y_pred = model.predict(x) #use it for predictions with either existing or new  data
print('predicted reponse:', y_pred, sep='\n')
#same as
y_pred = model.intercept_ + model.coef_ *x
print('predicted reponse:', y_pred, sep='\n')
#new data
x_new = np.arange(5).reshape((-1,1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)