import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

style.use('fivethirtyeight')

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?',-99999, inplace=True)

unif_cel_size = df['unif_cel_size']
#print (unif_cel_size)

marg_adhesion = df['marg_adhesion']
#print (marg_adhesion)

classif = df['class']
#print (classif)

vals = []

for i in range(0,len(classif)):
    vals.append([unif_cel_size[i],marg_adhesion[i],classif[i]])

# criar dicionário
# iterar sobre vals, se terminar em 2, adicionar à parte com 2, se terminar em 4, adicionar à parte maligno

for i in vals:
    if i[2]==2:

        plt.scatter(i[0],i[1],s=50,color='b')
    elif i[2]==4:
        plt.scatter(i[0],i[1],s=50,color='r')
plt.xlabel('uniform cell')
plt.ylabel('marginal adhesion')
plt.title('Fronteira de decisão para 2 variáveis')
plt.show()
