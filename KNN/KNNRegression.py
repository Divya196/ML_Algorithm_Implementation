import numpy as np
def sorted_distance(self, X_train, X_test):
        # We are setting a range of K values and calculating the RMSE for each of them. This way we can chose the optimal K value
        # Calculating the distance matrix using numpy broadcasting technique 
        #Sorting each data points of the distance matrix to reduce computational effort
        distance = np.sqrt(((X_train[:, :, None] - X_test[:, :, None].T) ** 2).sum(1))
        sorted_distance = np.argsort(distance, axis = 0) 
        return sorted_distance

#The knn function takes in the sorted distance and returns the RMSE of the 
def knn(self, X_train, X_test, y_train, y_test, sorted_distance, k):
    y_pred = np.zeros(y_test.shape)
    for row in range(len(X_test)):
        
        #Transforming the y_train values to adjust the scale. 
        y_pred[row] = y_train[sorted_distance[:,row][:k]].mean() * sigma_y + mu_y

    RMSE = np.sqrt(np.mean((y_test - y_pred)**2))
    return RMSE

def min_rmse_k_value(self, X_train, X_test, y_train, y_test, sorted_distance):
      k_list = [x for x in range(1,50,1)]
      #Storing the RMSE values in a list for each k value 
      rmse_list = []
      for i in k_list:
           rmse_list.append(knn(X_train,X_test,y_train,y_test,sorted_distance,i))
      #Finding the optimal K value
      min_rmse_k_value = k_list[rmse_list.index(min(rmse_list))]
      return min_rmse_k_value

class KNNRegression:
    def __init__(self, k, X_train, X_test, y_train, y_pred ):
        self.k = k
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_pred

    def fit(self, X, y):
        self.X_train = X 
        self.y_train =y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x , k):
       sorted_distance = sorted_distance( X_train, X_test)
       #RMSE = knn(X_train, X_test, y_train, y_test, sorted_distance, k)
       min_rmse_k_value= min_rmse_k_value(self, X_train, X_test, y_train, y_test, sorted_distance)
       return min_rmse_k_value

    
    











     