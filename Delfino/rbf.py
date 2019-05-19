import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


'''
material de consulta
Livro
Aulas no Youtube na Caltech do Abu-mostafa
Wikipedia
Post esclarecedor: http://www.jessebett.com/Radial-Basis-Function-USRA/
'''
n = 5000

# 8 mil pontos de -10 a 10 no eixo X
X = np.linspace(-10, 10, num=n)

#começo com uma lista, mas para usar numpy tem que virar array
lista_X = []

# fazer uma array of arrays
for i in X:
    lista_X.append([i])
array_lista_X=np.array([np.array(xi) for xi in lista_X])

X = array_lista_X

Y = (X)**2

centroides_num = 10 # numero de centros da RBF

# acho o k via random.choice, e não validação cruzada
index=np.random.choice(a=n,size=centroides_num) 

subsample=X[index,:] 

gamma = 0.5

kernel = np.exp(-gamma*euclidean_distances(X=X, Y=subsample,squared=True))
para = np.linalg.lstsq(kernel, Y)[0]

predict_Y = np.dot(kernel, para)

plt.plot(X, Y, 'r', label='Dados originais')
plt.plot(X, predict_Y, 'b', label='Após o data fit')
plt.legend()
plt.show()
