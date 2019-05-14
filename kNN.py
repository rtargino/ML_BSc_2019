import random as r
import numpy as np
import pandas as pd
import copy

class data():
    def __init__(self, array):
        self.array = array
        self.x = array[0:-1]
        self.y = array[-1]

    def distance(self, data):
        return np.dot(self.x,data.x)

    def near(self, list):
        list = copy.deepcopy(list)
        minim = list[0]
        dist_min = self.distance(list[0])
        savedi = 0
        for i in range(len(list)):
            if self.distance(list[i]) < dist_min:
                savedi = i
                minim = list[i]
                dist_min = self.distance(list[i])
        list.pop(savedi)
        return minim, list

    def knear(self, k, list, anslist=[]):
        if k == 0:
            return anslist
        else:
            ans, list = self.near(list)
            anslist.append(ans)
            return self.knear(k-1, list, anslist)

    def classify(self, k, list):
        try:
            return self.guess
        except:
            result = self.knear(k, list)
            counter = 0
            for data in result:
                counter += data.y
            return int(counter > k / 2)

class DataLearned():
    def __init__(self, list):
        """
        :type list: List
        """
        self.list = list

    def error_in(self):
        error = 0
        for elem in self.list:
            error += int(elem.classify() != elem.y)
        return error
