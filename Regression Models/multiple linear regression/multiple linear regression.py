import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import r2_score

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#  Handle categorical variables using dummy variables (OneHot Encoding)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')  # what kind of
# transformation we want to do and on which indexes, remainder(Keep the cols that are not transformed ?)

x = np.array(ct.fit_transform(x))  # expects a numpy array
print(x)
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# No need to apply feature scaling in multiple linear regression
# Dummy trapping do not have to be handled explicitly. The library will handle it.
# This library will also eliminate the non-significant features (low p values)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predict test set results
y_pred = regressor.predict(x_test)
print("R2 value: ", r2_score(y_test, y_pred))

np.set_printoptions(precision=2)
print(np.concatenate([y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)], axis=1))  # reshape(no of rows, no of cols)
# since we want to concatenate two vertical vectors, we pass 1 as the 2nd argument

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
# Dummy variables are always in the first column

# getting the multiple linear regression equation
print(regressor.coef_)
print(regressor.intercept_)

#  Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3+0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53
