from collections import Counter
import numpy as np
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
class KNN:
    def __init__(self,k=3):
        self.k=k
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    def predict(self,X):
        predicted_labels  = [self._predict(x) for x in X ]
        return np.array (predicted_labels)
    def _predict(self, x):
        #computing the distance 
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
        
        #getting the k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_lables = [self.y_train[i] for i in k_indices]

        # most common class 
        most_common  =Counter(k_nearest_lables).most_common(1)
        return most_common[0][0]

from sklearn.datasets import load_iris
# from KNN import *
from sklearn.model_selection import train_test_split
import numpy as np
iris = load_iris()
X,y= iris.data, iris.target

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=1234)

clf = KNN(k=5)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)

acc = np.sum(prediction == y_test)/ len(y_test)
print(acc)