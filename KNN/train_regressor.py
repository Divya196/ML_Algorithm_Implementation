import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNNRegression import KNNRegression
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

dataset = pd.read_csv(r'KNN\nba_2013.csv')
dataset = dataset.iloc[:,1:29]
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

train_split_percent = 0.7

size = X.shape[0]
X_train = X[:int(train_split_percent * size),:]
X_test = X[int(train_split_percent * size):,:]
y_train = y[:int(train_split_percent * size)]
y_test = y[int(train_split_percent * size):]


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
"""
#Standardizing the X_train and X_test daatsets
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu ) / sigma

#We use the same mean and SD as the one of X_train as we dont know the mean of X_test
X_test = (X_test - mu ) / sigma

#Standardizing the y_train data
mu_y = np.mean(y_train, 0)
sigma_y = np.std(y_train, 0, ddof = 0)

y_train = (y_train - mu_y ) / sigma_y

y_train = y_train.reshape(len(y_train),1)
y_test = y_test.reshape(len(y_test),1)

y_pred = np.zeros(y_test.shape)
y_train.shape, y_test.shape,y_pred.shape
"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)
#plt.figure()


#plt.figure()
#plt.scatter(X[:,1],X[:,2], c=y, cmap=cmap , edgecolor='k' ,s=20)
#plt.show()

regressor = KNNRegression(5,X_train,X_test,y_train,y_test)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)