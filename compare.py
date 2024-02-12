import pandas as pd
from utils import Log, plot_1v1_perf
from constants import UNIVARIATE_DATASET_NAMES_2018 as datasets

log = Log()

df = pd.read_csv("results-uea-avg-std.csv")


# remove std
def remove_parenthesis(value):
    return float(value.split("(")[0])

df.iloc[:, 1:] = df.iloc[:, 1:].applymap(remove_parenthesis)


df['Unnamed: 0'] = df['Unnamed: 0'].str.lower()
dataset_names = pd.Series(datasets, name="Unnamed: 0").str.lower()
df_copy = df.merge(dataset_names, on='Unnamed: 0', how='right')
df_copy.dropna(axis=0, inplace=True)

myresults = pd.read_csv("results.csv")
myresults['Unnamed: 0'] = myresults['Dataset'].str.lower()

# merge and remove unused columns
result = myresults.merge(df_copy,on="Unnamed: 0",how="right").drop(["Unnamed: 0", "Train Accuracy"], axis=1)
log.success(result)


for col in result.drop(["Dataset","Test Accuracy"], axis=1).columns:
    plot_1v1_perf(result,col,"Test Accuracy")