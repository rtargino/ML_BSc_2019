import random
from numpy import sign
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def gen_data (points = 100, x1_range = [0,10], x2_range = [0, 10], linear_separator = lambda x1, x2: sign(x1 - x2 + 1)):
    """
    Generate R2 random data linear separable.
    
    Parameters
    ----------
    points : int 
      number of points to be generate
    
    x1_range : list
      interval that x1 belongs
    
    x2_range : list
      interval that x2 belongs

    linear_separator : function
      a binary function that linearly separates the 
      data, that function need returns +1 or -1
    
    Returns
    -------
    negatives : list of binary tuples
      the data classified with -1 by the linear separator_function

    positives : list od binary tuples
      the data classified with +1 by the linear separator_function
    """
    
    positives_x1 = []
    positives_x2 = []
    negatives_x1 = []
    negatives_x2 = []
    
    for i in range(points):
        x1 = random.uniform(x1_range[0], x1_range[1])
        x2 = random.uniform(x2_range[0], x2_range[1])
        sign = linear_separator(x1, x2)
        
        if sign == 1:
            positives_x1.append(x)
            positives_x2.append(y)
        elif sign == -1:
            negatives_x1.append(x)
            negatives_x2.append(y)
    
    negatives = list(map(lambda x1, x2: (x1, x2), positives_x1, positives_x2))
    positives = list(map(lambda x1, x2: (x1, x2), negatives_x1, negatives_x2))
     
    return negatives , positives 


def find_separator(negative, positive, w = [1,1,1]):
    """
    Visualization of perceptron learning algorithm.

    Parameters
    ----------
    negatives : list of binary tuples
      the data classified with -1 by the linear separator_function

    positives : list od binary tuples
      the data classified with +1 by the linear separator_function
    
    """
    positives_x1 = list(map(lambda x: x[0], positive))
    positives_y = list(map(lambda x: x[1], positive))
    negatives_x = list(map(lambda x: x[0], negative))
    negatives_y = list(map(lambda x: x[1], negative))
    
    fig1 = plt.figure()
    
    points_p, points_n, line = plt.plot(positives_x1, positives_y, 'bx',\
                                        negatives_x, negatives_y, 'rx',\
                                        [], [])
    
    def update_frame(frame):
        
        misclassified = []
        nonlocal w
        nonlocal negative
        nonlocal positive
        for i in range(len(positive)):
            x = positives_x[i] 
            y = positives_y[i]
            
            if sign(w[0] * x + w[1] * y + w[2]) != 1:
                misclassified.append((x, y, 1))
                
        for i in range(len(negative)):
            x = negatives_x[i] 
            y = negatives_y[i]
            
            if sign(w[0] * x + w[1] * y + w[2]) != -1:
                misclassified.append((x, y, -1))

        if misclassified == []:

            return points_p, points_n, line

        else:
            target = random.choice(misclassified)
        
            w = list(map(lambda x, y: x + y * target[2], w, list(target[:2]) + [1]))
        
            line.set_data([0, 10], [(-w[2]) / w[1], (- w[2] - w[0] * 10) / w[1]])
        
            return points_p, points_n, line
    
    line_ani = animation.FuncAnimation(fig1, update_frame, 10, interval=500, blit=True)

    plt.show()


# Example:

positive, negative = gen_data(points=20)

find_separator(negative, positive)

