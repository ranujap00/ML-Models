import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#  Feature scaling the entire data set
sc = StandardScaler()  # range: -3 to +3
x_train = sc.fit_transform(x_train)  # 0,1,2 are dummy variables
x_test = sc.transform(x_test)

# scalar is fitted only to the training set to identify the scaling units.
# then it is applied to both training and test sets.
# test set is scales according to the model fitted to the training set.

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred_single = classifier.predict(sc.transform([[30, 87000]]))
# print(y_pred_single)

y_pred_test = classifier.predict(x_test)

np.set_printoptions(precision=2)
# print(np.concatenate([y_pred_test.reshape(len(y_pred_test), 1), y_test.reshape(len(y_test), 1)], axis=1))

# confusion matrix - shows how many correct and incorrect predictions are made.
# to evaluate the accuracy of a classification
confMatrix = confusion_matrix(y_test, y_pred_test)
print(confMatrix)
# class 0 - who did not buy SUV
# class 1 - wh did buy SUV

# 57 correct predictions from class 0
# 5 incorrect predictions from class 0 (5 predicted to have bought the SUV which is incorrect)
# 1 incorrect prediction from class 1
# 17 correct predictions from class 1

print(accuracy_score(y_test, y_pred_test))


# visualizing training set
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()