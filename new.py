from utils import read_dataset, plot

df = read_dataset("Coffee")

plot(df["Coffee"][0], df["Coffee"][1])

df = read_dataset("BirdChicken")

plot(df["BirdChicken"][0], df["BirdChicken"][1])