import random
from numpy import sign
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def read_data(filePath, number1, number2):

    negatives = []
    positives = []

    f = open(filePath, 'r')

    lines = f.readlines()

    for line in lines:
        line = line.split()
        if float(line[0]) == number1:
            positives.append((float(line[1]), float(line[2])))

        elif float(line[0]) == number2:
            negatives.append((float(line[1]), float(line[2])))

    return positives, negatives


def calcE(data1, data2, w):
    misclassified = []

    for i in range(len(data1)):
        x = data1[i][0]
        y = data1[i][1]

        if sign(w[0] * x + w[1] * y + w[2]) != 1:
            misclassified.append((x, y, 1))

    for i in range(len(data2)):
        x = data2[i][0]
        y = data2[i][1]

        if sign(w[0] * x + w[1] * y + w[2]) != -1:
            misclassified.append((x, y, -1))

    return misclassified


def pla(negative, positive, w=[1, 1, 1], max_iterations=1000):
    """
    Visualization of perceptron learning algorithm.

    Parameters
    ----------
    negatives : list of binary tuples
      the data classified with -1 by the linear separator_function

    positives : list od binary tuples
      the data classified with +1 by the linear separator_function

    w: list
      perceptron init value

    """

    # fig1 = plt.figure()

    misclassified = []
    k = 0

    while misclassified != [] and k != 0 and k < max_iterations:

        # E_in:
        misclassified = calcE(positive, negative, w)

        target = random.choice(misclassified)

        w = list(map(lambda x, y: x + y *
                     target[2], w, list(target[:2]) + [1]))

        k = k + 1

    return w


def pocket(negative, positive, w=[1, 1, 1], max_iterations=1000):
    """
    Visualization of perceptron learning algorithm.

    Parameters
    ----------
    negatives : list of binary tuples
      the data classified with -1 by the linear separator_function

    positives : list od binary tuples
      the data classified with +1 by the linear separator_function

    w: list
      perceptron init value

    """

    W = []
    Wscore = len(negative) + len(positive)
    # fig1 = plt.figure()

    misclassified = []
    k = 0

    while misclassified != [] and k != 0 and k < max_iterations:

        misclassified = calcE(positive, negative, w)

        target = random.choice(misclassified)

        if len(misclassifiedTrain) < Wscore or k == 0:
            W = w
            Wscore = len(misclassifiedTrain)

        w = list(map(lambda x, y: x + y *
                     target[2], w, list(target[:2]) + [1]))

        k = k + 1

    return W, Wscore


def plaVSpocket(trainFilePath, testFilePath, number1, number2, w=[1, 1, 1], max_iterations=1000):

    trainData1, trainData2 = read_data(trainFilePath, number1, number2)

    testData1, testData2 = read_data(testFilePath, number1, number2)

    W = []
    Wscore = len(trainData2) + len(trainData1)
    # fig1 = plt.figure()

    misclassifiedTrain = []
    k = 0
    Epla = []
    Epocket = []
    while (misclassifiedTrain != [] or k == 0) and k < max_iterations:

        # E_in_pla:
        misclassifiedTrain = calcE(trainData1, trainData2, w)

        # E_out_pla:
        misclassifiedTest = calcE(testData1, testData2, w)

        target = random.choice(misclassifiedTrain)

        if len(misclassifiedTrain) < Wscore or k == 0:
            W = w
            Wscore = len(misclassifiedTrain)

            Epocket.append((len(misclassifiedTrain), len(misclassifiedTest)))
        else:
            Epocket.append(Epocket[-1])

        Epla.append((len(misclassifiedTrain), len(misclassifiedTest)))

        k = k + 1
        if not (k == max_iterations or misclassifiedTrain == []):
            w = list(map(lambda x, y: x + y *
                         target[2], w, list(target[:2]) + [1]))

    fig1 = plt.figure()

    plt.plot([x[0] for x in trainData1], [x[1] for x in trainData1], 'bx',
             [x[0] for x in trainData2], [x[1] for x in trainData2], 'rx',
             [0, 0.6], [(-w[2]) / w[1], (- w[2] - w[0] * 0.6) / w[1]], 'r',
             [0, 0.6], [(-W[2]) / W[1], (- W[2] - W[0] * 0.6) / W[1]], 'g')

    fig2 = plt.figure()
    plt.plot([k for k in range(len(Epla))], Epla)

    fig3 = plt.figure()
    plt.plot([k for k in range(len(Epocket))], Epocket)

    plt.show()


if __name__ == '__main__':

    plaVSpocket("/home/hmuniz/FGV/ML_BSc_2019/features.train",
                "/home/hmuniz/FGV/ML_BSc_2019/features.test", 1, 5)
