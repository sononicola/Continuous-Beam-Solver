from continuous_beam_solver.solver import Solver
from continuous_beam_solver.span_beam import Span, Beam
from continuous_beam_solver.internal_forces import BendingMoment, Shear
from continuous_beam_solver.tables import Table
from continuous_beam_solver.global_variables import *

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

J = (0.3 * 0.5**3)/12 # m4
EJ =  31476*1000000*J/1000 # Mpa * m4 -> N*m2 -> kN*m2

#Campate 1,2,3
LOAD_S_A=114.972
LOAD_F_A=46.342
#Campata 4 con M4
LOAD_S_B=90.22675
LOAD_F_B=38.584
#Campate 5,6
LOAD_S_C=71.935
LOAD_F_C=32.67

c_1 = Span(lenght = 3.00, ej = EJ, q_max=LOAD_S_A, q_min=LOAD_F_A)
c_2 = Span(lenght = 4.50, ej = EJ, q_max=LOAD_S_A, q_min=LOAD_F_A)
c_3 = Span(lenght = 4.00, ej = EJ, q_max=LOAD_S_A, q_min=LOAD_F_A)
c_4 = Span(lenght = 5.00, ej = EJ, q_max=LOAD_S_B, q_min=LOAD_F_B)
c_5 = Span(lenght = 6.15, ej = EJ, q_max=LOAD_S_C, q_min=LOAD_F_C)
c_6 = Span(lenght = 4.00, ej = EJ, q_max=LOAD_S_C, q_min=LOAD_F_C)

trave = Beam(spans = [c_1, c_2, c_3, c_4, c_5, c_6], left_support="Simple", right_support="Fixed")

def run(beam: Beam):
    sol = Solver(beam)
    x = sol.generate_expanded_x_solutions()
    #print(f"x = {x}")
    r = sol.generate_R_solutions(x)
    #print(f"R = {r}")

    M = BendingMoment(beam, x, r)
    V = Shear(beam, x, r)
    
    #M.plot_span_Q_1(0)
    #M.plot_beam_Q_1()
    M.plot_inviluppo()
    plt.show()
    plt.close()

    

    #fig, ax = plt.subplots(1,1, figsize = (10, 5), tight_layout=True)
    #ax.invert_yaxis()
    #ax.fill_between(M.s_func, M.inviluppo()[0], color='r')
    #ax.fill_between(M.s_func, M.inviluppo()[1], color='b')
    #ax.axhline(0, color='grey', linewidth=2)
    #plt.show()

    # coordinates of inviluppo plot
    cords_x = M.s_func
    cords_y_pos, cords_y_neg = M.inviluppo()
    print(f"{cords_y_pos = }")
    print(f"{cords_y_neg = }")
    print(f"{cords_y_pos[-1] = }")
    print(f"{cords_y_pos[0] = }")
    print(f"{cords_y_pos[-2] = }")
    print(f"{cords_y_pos[1] = }")
    print(f"{cords_y_neg[-1] = }")
    print(f"{cords_y_neg[0] = }")
    print(f"{cords_y_neg[-2] = }")
    print(f"{cords_y_neg[1] = }")
    print(f"{cords_x[-1] = }")
    print(f"{cords_x[0] = }")
    print(len(cords_x))
    df_results_M = Table.create_dataframe(
        header=Table.make_header(len(trave.spans)),
        rows = Table.make_body(cords_x, cords_y_pos, cords_y_neg,trave.spans_cum_lenght()),
        index = ["s", "m_pos","m_neg"]
    )  
    print(df_results_M)
    print(cords_x)
    print(len(cords_x))

if __name__ == "__main__":
    run(trave)