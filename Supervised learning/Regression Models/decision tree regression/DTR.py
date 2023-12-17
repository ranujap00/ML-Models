import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# No need to apply feature scaling
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values  # first column is not needed (Range gives a 2D array)
y = dataset.iloc[:, -1].values


regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

#  not the best to use with a single feature data set

y_pred = regressor.predict([[6.5]])
print(y_pred)  # Not a good prediction

#  by increasing the degree, fitting of the curve can be increased
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
