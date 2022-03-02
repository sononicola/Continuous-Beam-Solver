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

    def make_body(cords_x, cords_y_pos, cords_y_neg, list_of_points, tol=0.001/2) -> list[list]:
        indexes_maxs = list_of_max_indexes(x = cords_x, y = cords_y_pos, list_of_points = list_of_points, tol=tol)
        indexes_mins = list_of_min_indexes(x = cords_x, y = cords_y_neg, list_of_points = list_of_points, tol=tol)
        print(indexes_maxs)
        print(indexes_mins)
        s = [cords_x[index] for index in indexes_maxs]
        s[-1] = s[-1] + 2*tol # due to np.arange aproximation, last term was not equal to total_lenght
        
        indexes_maxs[0] = 1 # for some reason cords_y_pos[0] and cords_y_neg[0] aren't correct, 
                            # so cords_y_pos[1] solves the problems

        body_table = [
                        s, 
                        [cords_y_pos[index] for index in indexes_maxs],
                        [cords_y_neg[index] for index in indexes_maxs]
                    ]
        return body_table

    def create_datafrate(header:list, rows:list[list], index:list) -> pd.DataFrame:
        return pd.DataFrame(columns=header, data=rows, index=index)



