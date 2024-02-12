from models import MLP
from utils import read_dataset, label_encoder

# Coffee dataset

df = read_dataset("Coffee")
X = df["Coffee"][0]
y = df["Coffee"][1]
model = MLP()
model.fit(X, y)


# y_pred_train = model.predict(X)

# accuracy_train = model.score(y_pred_train, y)

# y_pred_test = model.predict(df["Coffee"][2])

# accuracy_test = model.score(y_pred_test, df["Coffee"][3])

# print("Coffee train: ",accuracy_train)
# print("Coffee test: ",accuracy_test)

