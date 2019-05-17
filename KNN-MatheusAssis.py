'''
ALUNO:  Matheus Assis
CÓDIGO: K-Nearest Neighbors
'''

import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd

#função pré determinada
def function(position):
    x = position[0]
    return x ** 2

#função para determinar a distância euclidiana entre dois vetores de dimensão n
def euclidean_distance(position1, position2, dimension):
    distance = 0
    for i in range(dimension):
        distance += (position1[i] - position2[i]) ** 2
    return distance ** (1/2)

class KNN():
    def __init__(self, k, position, data_set):
        self.k         = k
        self.position  = position
        self.data_set  = data_set
        self.dimension = len(position[0]) 
        self.results   = list()
        
    def adjust_data(self):
        #ajustando os dados e gerando o "y" pela função estabelecida
        data_function             = [function(i[1]['Position']) for i in self.data_set.iterrows()]
        self.data_set['Function'] = data_function
        
    def get_distances(self):
        #determinando as distâncias de cada ponto para os que desejamos estimar
        for i in range(len(self.position)):
            data_distance                         = [euclidean_distance(self.position[i], j[1]['Position'], self.dimension) for j in self.data_set.iterrows()]
            self.data_set['Distance{}'.format(i)] = data_distance
    
    def get_neighbors(self):
        #escolhendo os k vizinhos mais próximos a partir da distância entre eles
        self.results, t = [[i, 0] for i in self.position], 0
        for i in self.position:
            k_nearest_neighbors  = self.data_set.sort_values(by='Distance{}'.format(t))[:self.k]
            mean_value           = np.mean([i[1]['Function'] for i in k_nearest_neighbors.iterrows()])
            self.results[t][1]   = mean_value
            t                   += 1
        
    def regression(self):
        #realizando cada um dos ajustes necessários
        self.adjust_data()
        self.get_distances()
        self.get_neighbors()
        
        #ajustando os dados a função pré estabelecida
        x_vars = [i[0] for i in self.data_set['Position']]
        y_vars = [i for i in self.data_set['Function']]
        
        #ajustando os resultados
        x_rslt = [i[0] for i in self.results]
        y_rslt = [i[1] for i in self.results]
        
        #plotando data set como bolinhas azuis e os resultados como bolinhas vermelhas
        plt.plot(x_vars, y_vars, 'bo')
        plt.plot(x_rslt, y_rslt, 'ro')
        plt.axis([0, len(self.data_set) + 5, min(self.data_set['Function']) - 10, max(self.data_set['Function']) + 10])
        plt.show()    
        
if __name__ == "__main__":
    #determinando o k, os vetores que queremos estimar pela regressão e o data set
    k = 3
    position = np.array([[np.random.choice(10)] for i in range(3)])
    aux_var  = np.array([[np.random.choice(10)] for i in range(10)])
    data_set = pd.DataFrame({'Position': list(aux_var)})
    
    #aplicando o modelo
    model = KNN(k, position, data_set)
    model.regression()