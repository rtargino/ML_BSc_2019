import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

pd.options.display.max_columns = 999

print (df.groupby('class').describe())

