import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import random
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?',-99999, inplace=True)

df.drop(['id'], 1, inplace=True)

# as features são tudo menos a coluna de classe
X = np.array(df.drop(['class'], 1))

y = np.array(df['class'])

# separar nos conjuntos de treino e de teste, para depois descobrir a acurácia
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

lista_valores_hiperparametros = random.sample(range(1,489),15)
print (lista_valores_hiperparametros)
lista_valores_hiperparametros.append(26)

lista_valores_hiperparametros.sort()
print (lista_valores_hiperparametros)

accuracy_list = []
for i in lista_valores_hiperparametros:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
    clf = neighbors.KNeighborsClassifier(i)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    accuracy_list.append(accuracy)

print (accuracy_list)

plt.plot(lista_valores_hiperparametros, accuracy_list , 'ro')
plt.axis([1,489,0,1])
plt.xlabel('Valores para o hiperparâmetro K')
plt.ylabel('Nível de Acurácia')
plt.title('Gráfico com o erro para cada um dos parâmetros')
plt.show()
