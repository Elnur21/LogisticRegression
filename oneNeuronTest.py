from models import LogisticRegression
from utils import read_dataset, label_encoder , plot_pie_chart

dataset = "Wine"

df = read_dataset(dataset)
X = df[dataset][0]
y = df[dataset][1]
model = LogisticRegression()
model.fit(X, y)


y_pred_train = model.predict(X)

accuracy_train = model.score(y_pred_train, y)

y_pred_test = model.predict(df[dataset][2])

accuracy_test = model.score(y_pred_test, df[dataset][3])

plot_pie_chart(y, y_pred_train, f"{dataset} Train Dataset")
plot_pie_chart(df[dataset][3], y_pred_test, f"{dataset} Test Dataset")

print(f"{dataset} train: ",accuracy_train)
print(f"{dataset} test: ",accuracy_test)

