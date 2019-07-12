import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.head())  
#df['unif_cel_size'].hist()
df.groupby('class')['unif_cel_size'].plot(kind='hist')

plt.xlabel("Uniform Cell Size (0-10)", fontdict=None, labelpad=None, )
plt.ylabel("Frequência do número de casos", fontdict=None, labelpad=None, )

plt.gca().legend(("2-benigno","4-maligno"))
plt.show()


