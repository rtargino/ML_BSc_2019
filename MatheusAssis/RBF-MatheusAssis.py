'''
ALUNO:  Matheus Assis
CÓDIGO: Radial Basis Function
'''

import numpy as np
import operator
import matplotlib.pyplot as plt

#função auxiliar para determinar a distância euclidiana entre dois pontos
def euclidean_distance(v1, v2):
    result = (v2 - v1) ** 2
    return result ** (1/2)
    
class RBF():
    def __init__(self, hidden_layers):
        #hidden_layers representa o número de variáveis na camada escondida
        self.hidden_layers = hidden_layers
        self.r       = 1.
        self.centers = None
        self.weights = None

    def define_centers(self, X):
        #definindo os centróides de maneira aleatória e sem ajustá-los
        random = np.random.choice(len(X), self.hidden_layers)
        centers = X[random]
        return centers

    def h(self, center, data_point):
        #cálculo que acontece em cada célula da camada escondida
        self.r = self.choose_r(center, self.centers, 2)
        return np.exp(-np.linalg.norm(data_point - center)**2/(self.r[0])**2)

    def H_aux(self, X):
        #passando por todas as células da camada escondida e calculando seu h(x)
        H = np.zeros((len(X), self.hidden_layers))
        for data_point_aux, data_point in enumerate(X):
            for center_aux, center in enumerate(self.centers):
                H[data_point_aux, center_aux] = self.h(center, data_point)
        return H

    def choose_r(self, center, centers, k):
        #escolhendo o r
        centers_aux, t = np.array([[0, 0] for i in range(len(centers))]), 0
        for cntr in centers:
            distance = euclidean_distance(center, cntr)
            centers_aux[t] = np.array([cntr, distance])
        nearest_neighbors = sorted(centers_aux, key=operator.itemgetter(1))[:k]
        r_aux = 0
        for cntr in nearest_neighbors:
            r_aux += euclidean_distance(center, cntr)
        return (r_aux/k) ** (1/2)

    def fit(self, X, Y):
        #ajustando os dados a uma função determinada
        self.centers = self.define_centers(X)
        H = self.H_aux(X)
        self.weights = np.dot(np.linalg.pinv(H), Y)

    def predict(self, X):
        #ajustando os dados ao RBF
        H = self.H_aux(X)
        predictions = np.dot(H, self.weights)
        return predictions

#EXAMPLE
        
if __name__ == "__main__":
    #determinando 100 valores de x entre 0 e 10 e ajustando-os na função 2 ** x_{i}
    x = np.linspace(0, 10, 100)
    y = np.exp2(x)
    
    #criando o modelo RBF
    model = RBF(hidden_shape=5)
    
    #ajustando os dados a uma função determinada
    model.fit(x, y)
    
    ##ajustando o modelo RBF aos dados escolhidos
    y_pred = model.predict(x)
    
    #plotando as curvas reais e ajustadas (azul e vermelho, respectivamente)
    plt.plot(x, y, 'b-', label='real')
    plt.plot(x, y_pred, 'r-', label='fit')
    plt.show()
