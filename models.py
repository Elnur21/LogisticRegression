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
    def __init__(self, learning_rate=0.01, num_iterations=1000, num_neurons=[3, 3]):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_neurons = num_neurons
        self.classifiers = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        num_samples, num_features = self.X.shape
        num_classes = len(np.unique(Y))
        self.classifiers = []

        num_layers = len(self.num_neurons)

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        # Initialize weights and biases for the first layer
        self.weights.append(np.random.randn(num_features, self.num_neurons[0]))
        self.biases.append(np.zeros(self.num_neurons[0]))

        # Initialize weights and biases for hidden layers and output layer
        for i in range(1, num_layers):
            self.weights.append(np.random.randn(self.num_neurons[i-1], self.num_neurons[i]))
            self.biases.append(np.zeros(self.num_neurons[i]))

        for class_label in range(num_classes):
            binary_labels = np.where(self.Y == class_label, 1, 0)

            for layer in range(num_layers):
                for _ in range(self.num_iterations):
                    # Forward pass
                    layer_input = self.X if layer == 0 else predicts
                    predicts = self.sigmoid(
                        np.dot(layer_input, self.weights[layer]) + self.biases[layer])

                    # Backpropagation
                    print(predicts.shape , binary_labels.shape)
                    if predicts.shape != binary_labels.shape:
                        predicts = predicts.T
                        
                    dw = (1 / num_samples) * np.dot(layer_input.T, (predicts - binary_labels))
                    db = (1 / num_samples) * np.sum(predicts - binary_labels)

                    # Update weights and biases
                    self.weights[layer] -= self.learning_rate * dw
                    self.biases[layer] -= self.learning_rate * db

            self.classifiers.append((self.weights, self.biases))
            log.info(self.classifiers)

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = []
            for layer_weights, layer_biases in self.classifiers:
                for weights, bias in zip(layer_weights, layer_biases):
                    score = np.dot(sample, weights) + bias
                    class_scores.append(score)
            predicted_class = np.argmax(class_scores)
            predictions.append(predicted_class)
        return np.array(predictions)

    def score(self, y_pred, y_true):
        return np.mean(y_true == y_pred) * 100
