import pandas as pd

from utils import find_local_max_xy, find_local_min_xy
import numpy as np

class Table:
    def make_header(n:int, string1:str = "A", string2:str = "C"):
        """
        Create the header of the table 

        Example: with n = 3 (== 3 spans) it returns: ['A1', 'C1', 'A2', 'C2', 'A3', 'C3', 'A4']

        By deafault: string1 = A == 'appoggio' in Italian == 'support' in English

        By deafault: string1 = C == 'campata' in Italian == 'span' in English 
        """
        header_odd = [f"{string1}{i}" for i in range(1,n+2)]
        header_even = [f"{string2}{i}" for i in range(1,n+1)]

        header = []
        for i in range(0,n):
            header.append(header_odd[i])
            header.append(header_even[i])

        header.append(header_odd[-1])

        return header

# TODO at the moment is strongly dependent from the two classes m and v:
    def make_principal_row(x: np.array, y:np.array, cum_lenghts:list):
        """Principal row is M or V and then associated row is V or M"""
        p_row = []
        DX = 0.
        s = [find_local_max_xy(x, y, cum_lenghts[i] , cum_lenghts[i+1], tol=0.001)[1] for i in range(0,4)]
        i = 0
        #s = find_local_max_xy(x, y, cum_lenghts[i] + DX , cum_lenghts[i+1] + DX)
        return s

    def list_of_max_indexes(x: np.array, y:np.array, list_of_points: list, tol:float) -> list:

        # Due to find_local_max_xy's tollerance value, we need to extend the lists because the last boundary exit from x list when checking np.where(b+tol)
        list_of_points = np.append(list_of_points, list_of_points[-1] + tol)
        x = np.append(x, x[-1] + tol)
        y = np.append(y, y[-1])

        # Maximum at supports:
        list_of_indexes_supports = [find_local_max_xy(x, y, list_of_points[i] , list_of_points[i], tol=tol)[0] for i in range(len(list_of_points)-1)]

        # Maximum in the middle of spans: (len -2 beacuse lists are n+1)
        list_of_indexes_spans = [find_local_max_xy(x, y, list_of_points[i] , list_of_points[i+1], tol=tol)[0] for i in range(0,len(list_of_points)-2)]

        list_of_indexes_all = list_of_indexes_supports + list_of_indexes_spans 
        list_of_indexes_all.sort()

        return list_of_indexes_all

    def list_of_min_indexes(x: np.array, y:np.array, list_of_points: list, tol:float) -> list:

        # Due to find_local_min_xy's tollerance value, we need to extend the lists because the last boundary exit from x list when checking np.where(b+tol)
        list_of_points = np.append(list_of_points, list_of_points[-1] + tol)
        x = np.append(x, x[-1] + tol)
        y = np.append(y, y[-1])

        # Minimum at supports:
        list_of_indexes_supports = [find_local_max_xy(x, y, list_of_points[i] , list_of_points[i], tol=tol)[0] for i in range(len(list_of_points)-1)]

        # Minimum in the middle of spans: (len -2 beacuse lists are n+1)
        list_of_indexes_spans = [find_local_min_xy(x, y, list_of_points[i] , list_of_points[i+1], tol=tol)[0] for i in range(0,len(list_of_points)-2)]

        list_of_indexes_all = list_of_indexes_supports + list_of_indexes_spans 
        list_of_indexes_all.sort()

        return list_of_indexes_all


# riga con la posizione dei valori

    def create_datafrate(header:list, rows:list[list], index:list):
        return pd.DataFrame(rows, columns=header, index=index)

df = Table.create_datafrate(
    header=Table.make_header(3),
    rows = [[1,2,3,4,5,6,7],[10,2,3,4,5,6,7],[1022,23,3,4,5,6,7]],
    index = ["s", "m_pos","m_neg"]
)
#print(df)

