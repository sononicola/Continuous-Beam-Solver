import pandas as pd

from maxs_mins import list_of_max_indexes, list_of_min_indexes
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

    


# riga con la posizione dei valori

    def create_datafrate(header:list, rows:list[list], index:list):
        return pd.DataFrame(rows, columns=header, index=index)

df = Table.create_datafrate(
    header=Table.make_header(3),
    rows = [[1,2,3,4,5,6,7],[10,2,3,4,5,6,7],[1022,23,3,4,5,6,7]],
    index = ["s", "m_pos","m_neg"]
)
#print(df)


