import numpy as np
from continuous_beam_solver.global_variables import *


def find_max_xy_all(x: list, y: list) -> list[tuple]:
    """
    Find the max y value of and x,y plot and the associated x. 
    
    It returns a list of tuples to prevent the case where there are more y max values (like in a costant y plot)  
    """
    if len(x) == len(y):
        indexes = [index for index, value in enumerate(y) if value == np.max(y)]
        return [(x[index], y[index]) for index in indexes]
    else:
        raise ValueError("x and y must have the same lenght")


def find_min_xy_all(x: list, y: list) -> list[tuple]:
    """
    Find the min y value of and x,y plot and the associated x. 

    It returns a list of tuples to prevent the case where there are more y min values (like in a costant y plot)
    """
    if len(x) == len(y):
        indexes = [index for index, value in enumerate(y) if value == np.min(y)]
        return [(x[index], y[index]) for index in indexes]
    else:
        raise ValueError("x and y must have the same lenght")


def find_max_xy(x: list, y: list) -> tuple:
    """
    Find the max y value of and x,y plot and the associated x. 

    Doesn't prevent the case where there are more y max values (like in a costant y plot). 
    In this case only the first occurrence is returned. See find_max_xy_all instead
    """
    index = np.argmax(y)
    return x[index], y[index]


def find_min_xy(x: list, y: list) -> tuple:
    """
    Find the min y value of and x,y plot and the associated x. 

    Doesn't prevent the case where there are more y min values (like in a costant y plot).
    In this case only the first occurrence is returned. See find_min_xy_all instead
    """
    index = np.argmin(y)
    return x[index], y[index]


def find_local_max_xy(
    x: np.array, y: np.array, a: float, b: float
) -> tuple:  # TODO sistemare il typing
    tol = ARANGE_STEP
    """
    Find the local max y value of and x,y plot and the associated x, corresponding to limits: x==a and x==b 
    Returns index, x(index), y(index) 

    x must be an np.array and should be an ordinated values array, like np.arange and np.linspace

    a: the initial value of x, not the corresponding index -> y[a_index : ]
    b: the final value of x, not the corresponding index
    tol: tollerance due to approximation on x (made with arange). tol should be equal of step/2. of arange function 

    Doesn't prevent the case where there are more y max values (like in a costant y plot). 
    In this case only the first occurrence is returned
    """
    a_index = np.where((x >= a - tol) & (x <= a + tol))[0][0]
    b_index = np.where((x >= b - tol) & (x <= b + tol))[0][0]
    # a_index = np.where((x >= a) & (x <= a + 20*tol))[0][0]
    # b_index = np.where((x >= b - 20*tol) & (x <= b))[0][0]
    index = np.argmax(
        y[a_index : b_index + 1]
    )  # this is the index of the sliced y, not of the original y array
    index = a_index + index  # so we need to add it to the index of the initial value a
    return (
        index,
        x[index],
        y[index],
    )  # so we need to add it to the index of the initial value a


def find_local_min_xy(x: np.array, y: np.array, a: float, b: float) -> tuple:
    tol = ARANGE_STEP
    """
    Find the local min y value of and x,y plot and the associated x, corresponding to limits: x==a and x==b.
    Returns index, x(index), y(index) 

    x must be an np.array and should be an ordinated values array, like np.arange and np.linspace

    a: the initial value of x, not the corresponding index -> y[a_index : ]
    b: the final value of x, not the corresponding index

    Doesn't prevent the case where there are more y min values (like in a costant y plot). 
    In this case only the first occurrence is returned
    """
    a_index = np.where((x >= a - tol) & (x <= a + tol))[0][0]
    b_index = np.where((x >= b - tol) & (x <= b + tol))[0][0]
    index = np.argmin(
        y[a_index : b_index + 1]
    )  # this is the index of the sliced y, not of the original y array
    index = a_index + index  # so we need to add it to the index of the initial value a
    return index, x[index], y[index]


def list_of_max_indexes(x: np.array, y: np.array, list_of_points: list) -> list:
    tol = ARANGE_STEP

    # Due to find_local_max_xy's tollerance value, we need to extend the lists because the last boundary exit from x list when checking np.where(b+tol)
    list_of_points = np.append(list_of_points, list_of_points[-1] + tol)
    x = np.append(x, x[-1] + tol)
    y = np.append(y, y[-1])

    # Maximum at supports:
    list_of_indexes_supports = [
        find_local_max_xy(x, y, list_of_points[i], list_of_points[i])[0]
        for i in range(len(list_of_points) - 1)
    ]
    list_of_indexes_supports[-1] = (
        list_of_indexes_supports[-1] - 1
    )  # last index corrisponds of total lenght in arange, but it's out of boundary for x and y
    # Maximum in the middle of spans: (len -2 because lists are n+1)
    list_of_indexes_spans = [
        find_local_max_xy(x, y, list_of_points[i], list_of_points[i + 1])[0]
        for i in range(0, len(list_of_points) - 2)
    ]

    list_of_indexes_all = list_of_indexes_supports + list_of_indexes_spans
    list_of_indexes_all.sort()

    return list_of_indexes_all


def list_of_min_indexes(x: np.array, y: np.array, list_of_points: list) -> list:
    tol = ARANGE_STEP

    # Due to find_local_min_xy's tollerance value, we need to extend the lists because the last boundary exit from x list when checking np.where(b+tol)
    list_of_points = np.append(list_of_points, list_of_points[-1] + tol)
    x = np.append(x, x[-1] + tol)
    y = np.append(y, y[-1])

    # Minimum at supports:
    list_of_indexes_supports = [
        find_local_min_xy(x, y, list_of_points[i], list_of_points[i])[0]
        for i in range(len(list_of_points) - 1)
    ]
    list_of_indexes_supports[-1] = (
        list_of_indexes_supports[-1] - 1
    )  # last index corrisponds of total lenght in arange, but it's out of boundary for x and y

    # Minimum in the middle of spans: (len -2 because lists are n+1)
    list_of_indexes_spans = [
        find_local_min_xy(x, y, list_of_points[i], list_of_points[i + 1])[0]
        for i in range(0, len(list_of_points) - 2)
    ]

    list_of_indexes_all = list_of_indexes_supports + list_of_indexes_spans
    list_of_indexes_all.sort()

    return list_of_indexes_all
