from models import LogisticRegression
from utils import read_dataset, label_encoder

# df = read_dataset("Coffee")
# X = df["Coffee"][0]
# y = df["Coffee"][1]
model = LogisticRegression()
# model.fit(X, y)


# y_pred_train = model.predict(X)

# accuracy_train = model.score(y_pred_train, y)

# y_pred_test = model.predict(df["Coffee"][2])

# accuracy_test = model.score(y_pred_test, df["Coffee"][3])

# print("Coffee train: ",accuracy_train)
# print("Coffee test: ",accuracy_test)

df = read_dataset("DistalPhalanxTW")
X = df["DistalPhalanxTW"][0]
y = df["DistalPhalanxTW"][1]
model.fit(X, label_encoder(y))


y_pred_train = model.predict(X)

accuracy_train = model.score(y_pred_train, label_encoder(y))

y_pred_test = model.predict(df["DistalPhalanxTW"][2])

accuracy_test = model.score(y_pred_test, label_encoder(df["DistalPhalanxTW"][3]))

print("DistalPhalanxTW train: ",accuracy_train)
print("DistalPhalanxTW test: ",accuracy_test)