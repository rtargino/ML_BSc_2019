import numpy as np
import matplotlib.pyplot as plt

def predict(x):
    x = np.array(x)
    w11 = [1,-1,0]
    w12 = [1,0,-1]
    w13 = [1,1,0]
    w14 = [1,0,1]
    w21 = [-3.5,1,1,1,1]
    W = np.array([[w11,w12,w13,w14],[w21]])
    s = np.array([np.array(x), np.zeros(4), np.zeros(1)])
    for i in range(1,3):
        for j in range(0,len(W[i-1])):
            s[i][j] = np.sign(np.dot(W[i-1][j],np.append([1],s[i-1])))
    return s[2][0]

def test():
    x = np.random.uniform(-2,2,200)
    y = np.random.uniform(-2,2,200)

    preds = np.array([predict(p) for p in zip(x,y)])

    plot = plt.scatter(x,y,c=preds)
    plot.axes.set_aspect('equal')
    plt.show()
 
if __name__ == "__main__":
    test()