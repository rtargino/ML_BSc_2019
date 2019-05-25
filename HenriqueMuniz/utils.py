import numpy


class BadCallError(ValueError):
    pass


def insert(list, new_item_with_distance):
    # Searching for the position
    index = 0
    for i in range(len(list)):
        if list[i][0] >= new_item_with_distance[0]:
            break
        else:
            index = index + 1

    return list[:index] + [new_item_with_distance] + list[index:]


def my_euclidian_dist(point1, point2):
    return numpy.linalg.norm(point1 - point2)
