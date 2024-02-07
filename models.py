import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate=learning_rate
        self.num_iterations=num_iterations
    
    def softmax(self, z):
        return np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)), keepdims=True)

    def fit(self,X, Y):
        self.X=X
        self.Y=Y
        m, features = self.X.shape

        self.weights = np.zeros(features)
        self.bias = 0

        for _i in range(self.num_iterations):

            predicts = self.softmax(np.dot(self.X, self.weights) + self.bias)

            dw = (1 / m) * np.dot(self.X.T, (predicts - self.Y))
            db = (1 / m) * np.sum(predicts - self.Y)


            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            



    def predict(self, X):
        return self.softmax(np.dot(X, self.weights) + self.bias)

    def score(self, y_pred, y_true):
        y_new = self.softmax(y_true)
        print(y_new)
        return np.mean(y_new == y_pred) *100



