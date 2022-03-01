from msilib.schema import Error
import numpy as np

def find_max_xy_all(x: list, y:list) -> list[tuple]:
    """
    Find the max y value of and x,y plot and the associated x. 
    
    It returns a list of tuples to prevent the case where there are more y max values (like in a costant y plot)  
    """
    if len(x) == len(y):
        indexes = [index for index, value in enumerate(y) if value == np.max(y)]
        return [(x[index] , y[index]) for index in indexes]
    else:
        raise ValueError("x and y must have the same lenght")


def find_min_xy_all(x: list, y:list) -> list[tuple]:
    """
    Find the min y value of and x,y plot and the associated x. 

    It returns a list of tuples to prevent the case where there are more y min values (like in a costant y plot)
    """
    if len(x) == len(y):
        indexes = [index for index, value in enumerate(y) if value == np.min(y)]
        return [(x[index] , y[index]) for index in indexes]
    else:
        raise ValueError("x and y must have the same lenght")

def find_max_xy(x: list, y:list) -> tuple:
    """
    Find the max y value of and x,y plot and the associated x. 

    Doesn't prevent the case where there are more y max values (like in a costant y plot). 
    In this case only the first occurrence is returned. See find_max_xy_all instead
    """
    index = np.argmax(y)
    return x[index], y[index]

def find_min_xy(x: list, y:list) -> tuple:
    """
    Find the min y value of and x,y plot and the associated x. 

    Doesn't prevent the case where there are more y min values (like in a costant y plot).
    In this case only the first occurrence is returned. See find_min_xy_all instead
    """
    index = np.argmin(y)
    return x[index], y[index]

def find_local_max_xy(x: np.array, y:np.array, a:float, b:float, tol:float) -> tuple: #TODO sistemare il typing
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
    index = np.argmax(y[a_index:b_index+1])     # this is the index of the sliced y, not of the original y array
    index = a_index + index                     # so we need to add it to the index of the initial value a
    return index, x[index], y[index]    # so we need to add it to the index of the initial value a

def find_local_min_xy(x: np.array, y:np.array, a:float, b:float, tol:float) -> tuple:
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
    index = np.argmin(y[a_index:b_index+1])     # this is the index of the sliced y, not of the original y array
    index = a_index + index                     # so we need to add it to the index of the initial value a
    return index, x[index], y[index]  

