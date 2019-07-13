import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

for i in df:
    print (i)
    print (df[str(i)].describe())
    #print (df[str(i)].kurtosis())
    print (" ")

""" 
    excluí o primeiro e último resultado do print pq era o ID e o output final
    sobre ser maligno ou benigno
"""

