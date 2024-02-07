import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate=learning_rate
        self.num_iterations=num_iterations
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self,X, Y):
        self.X=X
        self.Y=Y
        m, features = self.X.shape

        self.weights = np.zeros(features)
        self.bias = 0

        for _i in range(self.num_iterations):

            predicts = self.sigmoid(np.dot(self.X, self.weights) + self.bias)

            dw = (1 / m) * np.dot(self.X.T, (predicts - self.Y))
            db = (1 / m) * np.sum(predicts - self.Y)


            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            



    def predict(self, X):
        results = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return [1 if predict>0.5 else 0 for predict in results]

    def score(self, y_pred, y_true):
        return np.mean(y_true == y_pred) *100



