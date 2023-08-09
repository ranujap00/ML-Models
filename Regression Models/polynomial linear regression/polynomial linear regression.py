import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values  # first column is not needed
y = dataset.iloc[:, -1].values

# data is not split into test and training to get the maximum leverage to train the model
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#  by increasing the degree, fitting of the curve can be increased
poly_reg = PolynomialFeatures(degree=2)  # y = b0 + b1x1 + b2x1^2
x_poly = poly_reg.fit_transform(x)  # matrix with features x1 and x1^2

# polynomial regression coefficients are in the form of linear regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)  # this is the polynomial regression model

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# we can see linear regression is not a good fit for the given data

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#  by increasing the degree, fitting of the curve can be increased
# X_grid = np.arange(min(x), max(x), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(x, y, color = 'red')
# plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()


