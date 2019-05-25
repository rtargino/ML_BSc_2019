import utils
import numpy
import random


class knn_regression():

    def __init__(self, data, k=1, distanceFn=utils.my_euclidian_dist, condense=False, e=None):
        '''
        Parameters
        ----------
        k : The K-NN  according to the majority class among the k nearest data points to x
        data : A list of list, each list must be a
        '''
        def apply_condensation(data, e):
            def index_nearest_point(point, data):
                min_dist = float("inf")
                for i in range(len(data)):
                    dist = distanceFn(point, data[i][0])
                    if dist < min_dist:
                        min_dist = dist
                        index = i
                return index
            full_model = knn_regression(data, k=k)
            D = data.copy()
            S = []
            for i in range(k):
                element = D.pop(random.randrange(len(D)))
                S.append(element)

            ready = False
            while not ready:
                aux_model = knn_regression(S, k=k)
                misclassiy = []

                for i in range(len(data)):
                    diff_S_D = abs(full_model.classify(
                        data[i][0]) - aux_model.classify(data[i][0]))
                    if diff_S_D > e:
                        print(diff_S_D)
                        misclassiy.append(i)

                if len(misclassiy) > 0:
                    target = data[random.choice(misclassiy)]
                    new_element = D.pop(index_nearest_point(target[0], D))
                    S.append(new_element)
                else:
                    ready = True
            return S

        self.k = k
        self.distanceFn = distanceFn

        if condense:
            if e:
                self.data = apply_condensation(data, e)
            else:
                raise utils.BadCallError

        else:
            self.data = data

    def classify(self, x):
        aux = []
        for i in self.data:
            dist_i_x = self.distanceFn(i[:-1], x)
            if len(aux) < self.k:
                aux = utils.insert(aux, (dist_i_x, i))
            else:
                aux = utils.insert(aux, (dist_i_x, i))
                aux.pop()

        return numpy.mean([x[1][-1] for x in aux])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_data = [(numpy.array([x]), x ** 2)
                  for x in numpy.arange(0., 1., 0.1)]

    model = knn_regression(train_data, k=2)

    test_data = [numpy.array([random.uniform(0, 1)])
                 for i in range(4)]

    model_result = [(x, model.classify(x))
                    for x in test_data]

    plt.plot([train_data[i][0] for i in range(len(train_data))],
             [train_data[i][1] for i in range(len(train_data))],
             "bo",
             [model_result[i][0] for i in range(len(model_result))],
             [model_result[i][1] for i in range(len(model_result))],
             "ro")

    plt.show()

    train_data = [(numpy.array(x), x ** 2) for x in numpy.arange(0., 1., 0.02)]
    model = knn_regression(train_data, k=1, condense=True, e=0.1)

    plt.plot([train_data[i][0] for i in range(len(train_data))],
             [train_data[i][1] for i in range(len(train_data))],
             "ro",
             [model.data[i][0] for i in range(len(model.data))],
             [model.data[i][1] for i in range(len(model.data))],
             "bo")

    plt.show()
