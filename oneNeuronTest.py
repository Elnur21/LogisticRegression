from models import LogisticRegression
import numpy as np


num_samples = 10

# Number of features
num_features = 5

# Number of classes
num_classes = 5

# Generate random features
X = np.random.rand(num_samples, num_features)

# Generate random labels
y = np.random.randint(0, num_classes, size=num_samples)
model = LogisticRegression()
model.fit(X, y)


y_pred_train = model.predict(X)

accuracy = model.score(y_pred_train, y)

print(accuracy)