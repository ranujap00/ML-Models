import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values  # first column is not needed (Range gives a 2D array)
y = dataset.iloc[:, -1].values

#  y is a row vector. We have to convert it to a column vector
y = y.reshape(len(y), 1)  # gives a column 2D array

#  must undergo feature scaling as there is no regression equation and coefficients to balance the values
scLevel = StandardScaler()  # range: -3 to +3
x = scLevel.fit_transform(x)

#  since Level and salary does not have the same mean and SD, two SC objects should be created
scSalary = StandardScaler()
y = scSalary.fit_transform(y)

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# model is trained using scaled values. Therefore, have to pass scaled values for x.
y_pred = regressor.predict(scLevel.transform([[6.5]])).reshape(-1, 1)
y_pred = scSalary.inverse_transform(y_pred)  # convert from scaled to original form

print(y_pred)

plt.scatter(scLevel.inverse_transform(x), scSalary.inverse_transform(y), color='red')  # Get original values
plt.plot(scLevel.inverse_transform(x), scSalary.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')  # x is already transformed
plt.title('Truth or Bluff (SVR regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
