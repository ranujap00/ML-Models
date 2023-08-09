import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# x,y are matrices / numpy arrays

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=0)  # random_state=1 means same split everytime

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting test values
y_pred = regressor.predict(x_test)
print("R2 Value: ", r2_score(y_test, y_pred))

# Visualize real salaries vs predicted salaries
plt.scatter(x_train, y_train, color='red')  # actual values
plt.plot(x_train, regressor.predict(x_train), color='blue')  # best fit line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')  # best fit line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
