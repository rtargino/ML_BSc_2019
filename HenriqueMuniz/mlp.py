import numpy
import random

w1 = numpy.array([1, 0, -1])
w2 = numpy.array([1, 0, 1])
w3 = numpy.array([1, -1, 0])
w4 = numpy.array([1, 1, 0])
ws = [w1, w2, w3, w4]


def classify(x):
    # hidden layer

    hidden = [1]
    x_aux = numpy.array([1] + x)

    for wi in ws:
        hidden.append(numpy.sign(x_aux @ wi))

    # output layer (nothing more than a single AND)

    return numpy.sign(sum(hidden) - 3.5)


def gen_data(N):
    '''
    N random points insie of [-2:2]x[-2:2] square
    '''
    points = []

    for i in range(N):
        points.append([random.uniform(-2, 2), random.uniform(-2, 2)])

    return points


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    inside_points = []
    outside_points = []

    for i in gen_data(10000):
        value = classify(i)
        if value == 1:
            inside_points.append(i)
        else:
            outside_points.append(i)

    plt.plot([x[0] for x in inside_points],
             [x[1] for x in inside_points],
             "bo",
             [x[0] for x in outside_points],
             [x[1] for x in outside_points],
             "ro")
    plt.show()
