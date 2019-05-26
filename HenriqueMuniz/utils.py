import numpy
import random
import functools


class BadCallError(ValueError):
    pass


'''
function binary_search(A, n, T):
    L := 0
    R := n − 1
    while L <= R:
        m := floor((L + R) / 2)
        if A[m] < T:
            L := m + 1
        else if A[m] > T:
            R := m - 1
        else:
            return m
    return unsuccessful
'''


def identity(x):
    return x


def insert(arr, new_item_with_distance, acessorFn=identity):
    def search(arr, val, start, end, acessorFn=identity):
        if end - start == 1:
            if acessorFn(arr[end]) < val:
                return end + 1
            else:
                return start + 1

        if start == end:
            if acessorFn(arr[start]) > val:
                return start
            else:
                return start + 1

        if start > end:
            return start

        mid = int(numpy.floor((start + end) / 2))

        if acessorFn(arr[mid]) < val:
            return search(arr, val, mid, end, acessorFn=acessorFn)
        elif acessorFn(arr[mid]) > val:
            return search(arr, val, start, mid, acessorFn=acessorFn)
        else:
            return mid

    # Searching for the position
    index = search(arr, acessorFn(new_item_with_distance), 0, len(
        arr) - 1, acessorFn=acessorFn)
    arr.insert(index, new_item_with_distance)
    return arr


def my_euclidian_dist(point1, point2):
    return numpy.linalg.norm(point1 - point2)


def greedy_initialization_lloyd(data, k):
    aux_data = data.copy()
    x = aux_data.pop(random.randrange(len(aux_data)))[0]
    result = [x]

    for i in range(k - 1):
        max_dist = 0
        new_point_index = 0

        for j in range(len(aux_data)):
            dist = 0
            for point in result:
                dist = dist + my_euclidian_dist(point, aux_data[j][0])

            if dist > max_dist:
                max_dist = dist
                new_point_index = j

        result.append(aux_data.pop(new_point_index)[0])

    return result
