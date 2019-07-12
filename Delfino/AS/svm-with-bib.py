import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

# no arquivo de dados são 11 variáveis, 9 independetes, 1 dependente e 1 primary key
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# para dados faltantes
df.replace('?',-99999, inplace=True)

# tira a coluna de id pq ela não é nem variável dependente e nem independente, só primary key
df.drop(['id'], 1, inplace=True)

# as features são tudo menos a coluna de classe, a id já foi excluída
X = np.array(df.drop(['class'], 1))

# o Y na minha supervised learning é a class
y = np.array(df['class'])

# separar nos conjuntos de treino e de teste, para depois descobrir a acurácia
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

# poderia ser 'linear', 'poly', 'sigmoid'. Pelo que vimos em sala, usar 'rbf' 
clf = svm.SVC(kernel='rbf')

# inserir no objeto o treinamento
clf.fit(X_train, y_train)

# fazer o teste para saber a acurácia
accuracy = clf.score(X_test, y_test)

print("accuracy: ", accuracy)
print("test error: ", 1-accuracy)
print ("trainning error: ", 1-clf.score(X_train,y_train))

#scores = cross_validation.cross_val_score(clf,X,cv=5)
#print ("cross-validation error", scores)

'''
#exemplo pra prever se a célula é benigna ou maligna
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print("predição: ",prediction)
'''
