from utils import read_dataset, plot

df = read_dataset("Worms")

plot(df["Worms"][0], df["Worms"][1])