import numpy as np
from functools import reduce

class Three_Layer_Perceptron():
    def __init__(self, *args):
        self.perceptrons = args

    def And(self, x):
        return reduce(lambda x,y: x and y, [p.hip(x) for p in self.perceptrons])



class Perceptron():
    def __init__(self, size):
        self.size = size
        self.w = np.ones(size+1)

    def hip(self, x):
        return np.sign(np.dot(self.w,np.insert(x,0,1)))

    # def converged(self, data, ans):
    #     if data.shape == (1,self.size):
    #         if self.hip(data) == ans:
    #             return True
    #         else:
    #             self.misclassified = data
    #             return False
    #
    # def train(self, ans, misclassifed):
    #     if self.hip(misclassifed) != ans:
    #         self.w = self.w + ans * misclassifed
    #     else:
    #         raise Exception("Non consistent")

