import numpy as np
from utils import Log

log = Log()


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=5000, batch_size=16):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size=batch_size
        self.classifiers = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        num_samples, num_features = self.X.shape
        num_classes = len(np.unique(Y))
        self.classifiers = []

        for class_label in range(num_classes):
            binary_labels = np.where(self.Y == class_label, 1, 0)

            self.weights = np.zeros(num_features)
            self.bias = 0

            for _ in range(self.num_iterations):
                for i in range(0, num_samples, self.batch_size):
                    predicts = self.sigmoid(
                        np.dot(self.X[i:i+self.batch_size], self.weights) + self.bias)

                    dw = (1 / num_samples) * \
                        np.dot(self.X[i:i+self.batch_size].T, (predicts - binary_labels[i:i+self.batch_size]))
                    db = (1 / num_samples) * np.sum(predicts - binary_labels[i:i+self.batch_size])

                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db

            self.classifiers.append((self.weights, self.bias))

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = []
            for self.weights, self.bias in self.classifiers:
                score = np.dot(sample, self.weights) + self.bias
                class_scores.append(score)
            predicted_class = np.argmax(class_scores)
            predictions.append(predicted_class)
        return np.array(predictions)

    def score(self, y_pred, y_true):
        return np.mean(y_true == y_pred) * 100


class MLP:
    def __init__(self, learning_rate=0.01, num_iterations=1000, num_neurons=3):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_neurons = num_neurons
        self.classifiers = []

    def sigmoid(self, z):
        z2 = np.max(0, z)
        return 1 / (1 + np.exp(-z2))

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        num_samples, num_features = self.X.shape
        num_classes = len(np.unique(Y))
        self.classifiers = []

        for class_label in range(num_classes):
            binary_labels = np.where(self.Y == class_label, 1, 0)

            self.weights = np.zeros((self.num_neurons, num_features))
            self.bias = np.zeros(self.num_neurons)

            for i in range(self.num_neurons):
                for _ in range(self.num_iterations):
                    predicts = self.sigmoid(
                        np.dot(self.X, self.weights[i]) + self.bias[i])

                    dw = (1 / num_samples) * \
                        np.dot(self.X.T, (predicts - binary_labels))
                    db = (1 / num_samples) * np.sum(predicts - binary_labels)

                    self.weights[i] -= self.learning_rate * dw
                    self.bias[i] -= self.learning_rate * db

            self.classifiers.append((self.weights, self.bias))
            log.info(self.classifiers)

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = []
            for self.weights, self.bias in self.classifiers:
                score = np.dot(sample, self.weights) + self.bias
                class_scores.append(score)
            predicted_class = np.argmax(class_scores)
            predictions.append(predicted_class)
        return np.array(predictions)

    def score(self, y_pred, y_true):
        return np.mean(y_true == y_pred) * 100
