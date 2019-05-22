import numpy as np
import random
import matplotlib.pyplot as plt

# definindo os pesos arbitrariamente
w_1 = np.array([1,0,-1]) 
w_2 = np.array([1,0,1])
w_3 = np.array([1,-1,0])
w_4 = np.array([1,1,0])
    
def multi_perceptron(point):
    
    vector = np.array([1, point[0], point[1]])
   
    weights = [w_1,w_2,w_3,w_4]
    
    matrix_multiplication = [ ]

    for i in weights:
        matrix_multiplication.append(np.sign(np.matmul(i.T,vector)))        
    
    total = sum(matrix_multiplication) 

    point_sign_class = np.sign(-3.5+total)
   
    return point_sign_class

x = np.random.uniform(-3,3,650)
y = np.random.uniform(-3,3,650)

pontos = []

for i in range(0,len(x)):
    pontos.append((x[i],y[i]))

rectangular_class = []

for ponto in pontos:
    rectangular_class.append(multi_perceptron(ponto))

plt.scatter(x,y,c=rectangular_class, cmap="bwr_r")
plt.show()

