import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# no arquivo de dados são 11 variáveis, 9 independetes, 1 dependente e 1 é só primary-key
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# para dados faltantes
df.replace('?',-99999, inplace=True)

# tira a coluna de id pq ela não é nem variável dependente e nem independente
df.drop(['id'], 1, inplace=True)

# as features são tudo menos a coluna de classe, a id já foi excluída
X = np.array(df.drop(['class'], 1))

# o Y na minha supervised learning é a class
y = np.array(df['class'])

# separar nos conjuntos de treino e de teste, para depois descobrir a acurácia
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

k_values = list(range(1,51))


for i in k_values:

    # defini o hiperparâmetro como k=i, 26 está dentro do intervalo de iteração 
    clf = neighbors.KNeighborsClassifier(i)

    # inserir no objeto o treinamento
    clf.fit(X_train, y_train)

    # fazer o teste para saber a acurácia
    accuracy = clf.score(X_test, y_test)
    
    print("acurácia para o hiperparâmatro k=", i, ": ", accuracy)
    #print("erro no teste: ", 1-accuracy)
