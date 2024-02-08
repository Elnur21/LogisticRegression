import numpy as np

def log(input):
    print(input)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.classifiers = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        num_samples, num_features = self.X.shape
        num_classes = len(np.unique(Y))

        for class_label in range(num_classes):
            # Convert labels to binary labels (0 or 1)
            binary_labels = np.where(self.Y == class_label, 1, 0)

            # Initialize weights and bias for this class
            weights = np.zeros(num_features)
            bias = 0

            # Gradient Descent
            for _ in range(self.num_iterations):
                predicts = self.sigmoid(np.dot(self.X, weights) + bias)

                dw = (1 / num_samples) * np.dot(self.X.T, (predicts - binary_labels))
                db = (1 / num_samples) * np.sum(predicts - binary_labels)

                weights -= self.learning_rate * dw
                bias -= self.learning_rate * db

            # Store the trained classifier
            self.classifiers.append((weights, bias))

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = []
            for weights, bias in self.classifiers:
                # Compute score for each class
                score = np.dot(sample, weights) + bias
                class_scores.append(score)
            # Predict the class with the highest score
            predicted_class = np.argmax(class_scores)
            predictions.append(predicted_class)
        return np.array(predictions)

    def score(self, y_pred, y_true):
        return np.mean(y_true == y_pred) * 100


