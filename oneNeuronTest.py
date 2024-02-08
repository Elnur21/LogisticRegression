from models import LogisticRegression
import numpy as np
from utils import read_dataset

df = read_dataset("Coffee")
X = df["Coffee"][0]
y = df["Coffee"][1]
model = LogisticRegression()
model.fit(X, y)


y_pred_train = model.predict(X)

accuracy_train = model.score(y_pred_train, y)

y_pred_test = model.predict(df["Coffee"][2])

accuracy_test = model.score(y_pred_test, df["Coffee"][3])

print("Coffee train: ",accuracy_train)
print("Coffee test: ",accuracy_test)

df = read_dataset("Computers")
X = df["Computers"][0]
y = df["Computers"][1]
model.fit(X, y)


y_pred_train = model.predict(X)

accuracy_train = model.score(y_pred_train, y)

y_pred_test = model.predict(df["Computers"][2])

accuracy_test = model.score(y_pred_test, df["Computers"][3])

print("Computers train: ",accuracy_train)
print("Computers test: ",accuracy_test)