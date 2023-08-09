import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder  # to encode yes/no to 0/1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Data.csv')
# independent variables (features)
x = dataset.iloc[:, :-1].values  # locate index

# dependent variable
y = dataset.iloc[:, -1].values  # locate index

# Handle missing data (replace the missing with the avg of the column)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])  # returns the updated columns

#  Handle categorical variables using dummy variables (OneHot Encoding)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # what kind of
# transformation we want to do and on which indexes, remainder(Keep the cols that are not transformed ?)

x = np.array(ct.fit_transform(x))  # expects a numpy array

le = LabelEncoder()  # when only 2 classes are present
y = le.fit_transform(y)

# Feature scaling should be applied after the data set is split into training and data
# Splitting the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# x_train and y_train are corresponding to each other
print(x_train)
print(x_test)
print(y_train)
print(y_test)


# Feature Scaling
# Normalization is recommended when most of the features follow a normal distribution
# Standardization works all the time. Therefore, better to go with this.
# Feature scaling is not applied to dummy variables

sc = StandardScaler()  # range: -3 to +3
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])  # 0,1,2 are dummy variables
x_test[:, 3:] = sc.transform(x_test[:, 3:])

# fit - gives values for mean and sd
# transform - apply the formula to get the result

print(x_train)
print(x_test)
