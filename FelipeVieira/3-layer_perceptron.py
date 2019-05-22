import random as rd
import numpy as np
import matplotlib.pyplot as plt

def sign(r):
    if r < 0:
        return -1
    else:
        return 1
    
def ands(*s):
    hs = np.concatenate([np.array(s),np.ones(1)])
    ws = np.concatenate([np.ones(len(s)),np.array([-len(s)+0.5])])
    return sign(np.dot(hs,ws))

def multi_layer_perceptron_square(x):
    x = np.array(x)
    w1 = np.array((1,0,-1))
    w2 = np.array((1,0,1))
    w3 = np.array((1,1,0))
    w4 = np.array((1,-1,0))
    h1 = sign(np.dot(w1,x))
    h2 = sign(np.dot(w2,x))
    h3 = sign(np.dot(w3,x))
    h4 = sign(np.dot(w4,x))
    return ands(h1,h2,h3,h4)

def test(n):
    x = np.random.uniform(-2,2,n)
    y = np.random.uniform(-2,2,n)
    inputs  = list(zip(np.ones(n),x,y))
    predicts = [multi_layer_perceptron_square(x) for x in inputs]
    reals = [1 if (x[1]**2+x[2]**2)<1 else -1 for x in inputs]
    errors = [abs(predicts[i]-reals[i])/2 for i in range(len(inputs))]
    E_in = sum(errors)/len(inputs)
    print('E_in:',E_in)
    for i in range(n):
        if predicts[i] == reals[i] == 1:
            plt.plot(x[i],y[i],'bo')
        elif predicts[i] == reals[i] == -1:
            plt.plot(x[i],y[i],'bx')
        elif predicts[i] != reals[i] == 1:
            plt.plot(x[i],y[i],'rx')
        else:
            plt.plot(x[i],y[i],'ro')
    plt.gcf().gca().add_artist(plt.Circle((0,0),1))
    plt.show()

