import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNNClassifier import KNNClassifier
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

dataset = pd.read_csv('KNN\Social_Network_Ads.csv')
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#plt.figure()
#plt.scatter(X[:,1],X[:,2], c=y, cmap=cmap , edgecolor='k' ,s=20)
#plt.show()

classifier = KNN(k=5)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)