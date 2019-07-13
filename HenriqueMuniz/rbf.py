import utils
import numpy
import random


class RBFNetwork():
    def __init__(self, data, k, r, kernelFn):
        self.data = data
        self.centers = [x[0] for x in random.choices(data, k=k)]
        # self.centers = utils.greedy_initialization_lloyd(
        #     data, k, utils.my_euclidian_dist)
        self.kernelFn = kernelFn
        self.k = k
        self.r = r
        self.w = self._calculate_w()

    def _calculate_w(self):
        N = len(self.data)
        Z = numpy.empty((N, self.k + 1))
        y = numpy.array([x[1] for x in self.data])

        for n in range(N):
            Z[n][0] = 1
            for j in range(self.k):
                Z[n][j + 1] = self.kernelFn(
                    utils.my_euclidian_dist(self.data[n][0], self.centers[j]) / self.r)

        return numpy.linalg.pinv(Z).dot(y)

    def classify(self, x):
        result = self.w[0]

        for i in range(len(self.centers)):
            result = result + \
                self.w[i + 1] * \
                self.kernelFn(utils.my_euclidian_dist(
                    self.centers[i], x) / self.r)
        return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_data = [(numpy.array([x]), numpy.sin(x))
                  for x in numpy.arange(0., 10., 0.1)]

    k = 20
    r = 1 / k

    def kernel(x):
        return numpy.exp(- (0.5 * x ** 2))

    model = RBFNetwork(train_data, k, r, kernel)

    model_result = [(x[0], model.classify(x[0]))
                    for x in train_data]

    plt.plot([train_data[i][0] for i in range(len(train_data))],
             [train_data[i][1] for i in range(len(train_data))],
             "b-",
             [model_result[i][0] for i in range(len(model_result))],
             [model_result[i][1] for i in range(len(model_result))],
             "r-",
             [x for x in model.centers],
             [0 for x in model.centers],
             "go")

    plt.show()
