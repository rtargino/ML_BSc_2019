import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.head())  
#df['unif_cel_size'].hist()
df.groupby('class')['unif_cel_size'].plot(kind='hist', color="black")

plt.xlabel("Uniform Cell Size (0-10)", fontdict=None, labelpad=None, )
plt.ylabel("Frequência do número de casos", fontdict=None, labelpad=None, )
x = [1,2,3,4,5,6,7,8,9,10]
plt.xticks(x)
plt.show()


