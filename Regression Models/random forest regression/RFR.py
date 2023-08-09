import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# better adapted to high dimensional data set
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values  # first column is not needed (Range gives a 2D array)
y = dataset.iloc[:, -1].values

# random_state = 0 t0 fix the seed to get the same output everytime we run it
regressor = RandomForestRegressor(n_estimators=10, random_state=0)  # 10 trees
regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])
print(y_pred)

X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


