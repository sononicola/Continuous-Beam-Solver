from continuous_beam_solver.solver import Solver
from continuous_beam_solver.span_beam import Span, Beam
from continuous_beam_solver.internal_forces import BendingMoment, Shear
from continuous_beam_solver.tables import Table
from continuous_beam_solver.global_variables import *

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#plt.style.use('science')
#plt.style.context(['science','no-latex']) # If don't have latex de-comment

EXPORT_FOLDER = "export/"
LOAD_TYPE = "ULS" # "ULS" , "SLS_CHAR", "SLS_FREQ", "SLS_QP"

J = (0.3 * 0.5**3)/12 # m4
EJ =  31476*1000000*J/1000 # Mpa * m4 -> N*m2 -> kN*m2

#TODO da mettere nel main: for load in ["ULS" , "SLS_CHAR", "SLS_FREQ", "SLS_QP"]: LOAD_TYPE == load

if LOAD_TYPE == "ULS":
    #Campate 1,2,3
    LOAD_S_A=114.972
    LOAD_F_A=46.342
    #Campata 4 con M4
    LOAD_S_B=90.22675
    LOAD_F_B=38.584
    #Campate 5,6
    LOAD_S_C=71.935
    LOAD_F_C=32.67
elif LOAD_TYPE == "SLS_CHAR":
    #Campate 1,2,3
    LOAD_S_A=79.81
    LOAD_F_A=51.557
    #Campata 4 con M4
    LOAD_S_B=62.8
    LOAD_F_B=43.287
    #Campate 5,6
    LOAD_S_C=50.04
    LOAD_F_C=36.945
elif LOAD_TYPE == "SLS_FREQ":
    #Campate 1,2,3
    LOAD_S_A=63.24
    LOAD_F_A=51.903
    #Campata 4 con M4
    LOAD_S_B=51.67
    LOAD_F_B=43.489
    #Campate 5,6
    LOAD_S_C=50.04
    LOAD_F_C=37.0689
elif LOAD_TYPE == "SLS_QP":
    #Campate 1,2,3
    LOAD_S_A=58.74
    LOAD_F_A=51.99
    #Campata 4 con M4
    LOAD_S_B=48.42
    LOAD_F_B=43.54
    #Campate 5,6
    LOAD_S_C=40.55
    LOAD_F_C=37.10

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

# DECOMMENTARE CIO CHE SERVE:
    #M.plot_span_Q_1(0)
    #M.plot_beam_Q_1()
    #M.plot_inviluppo() #_trasposed()
    #plt.savefig(EXPORT_FOLDER+LOAD_TYPE+"_M.pdf")
    #plt.show()
    #plt.close()
    
    #V.plot_inviluppo()
    #plt.savefig(EXPORT_FOLDER+LOAD_TYPE+"_V.pdf")
    #plt.show()
    #plt.close()
    

    

    #fig, ax = plt.subplots(1,1, figsize = (10, 5), tight_layout=True)
    #ax.invert_yaxis()
    #ax.fill_between(M.s_func, M.inviluppo()[0], color='r')
    #ax.fill_between(M.s_func, M.inviluppo()[1], color='b')
    #ax.axhline(0, color='grey', linewidth=2)
    #plt.show()



    # coordinates of inviluppo plot
    M_cords_x = M.s_func
    M_cords_y_pos, M_cords_y_neg = M.inviluppo()

    M.plot_inviluppo_trasposed(
        list_of_points=Table.make_body(M_cords_x, M_cords_y_pos, M_cords_y_neg,trave.spans_cum_lenght())[0],
        delta=0.5) 


    #plt.savefig(EXPORT_FOLDER+LOAD_TYPE+"_M.pdf")
    plt.show()
    plt.close()

    #fig, ax = plt.subplots(1,1, figsize = (16, 10), tight_layout=True)
    #ax.invert_yaxis()
    #ax.fill_between(s_t, M_cords_y_pos)
    #plt.show()
    #plt.close()
    
    quit()
    print(f"{M_cords_y_pos = }")
    print(f"{M_cords_y_neg = }")
    print(f"{M_cords_y_pos[-1] = }")
    print(f"{M_cords_y_pos[0] = }")
    print(f"{M_cords_y_pos[-2] = }")
    print(f"{M_cords_y_pos[1] = }")
    print(f"{M_cords_y_neg[-1] = }")
    print(f"{M_cords_y_neg[0] = }")
    print(f"{M_cords_y_neg[-2] = }")
    print(f"{M_cords_y_neg[1] = }")
    print(f"{M_cords_x[-1] = }")
    print(f"{M_cords_x[0] = }")
    print(len(M_cords_x))
    M_df_results = Table.create_dataframe(
        header=Table.make_header(len(trave.spans)),
        rows = Table.make_body(M_cords_x, M_cords_y_pos, M_cords_y_neg,trave.spans_cum_lenght()),
        index = ["s", "M_neg","M_pos"]
    )  
    print(M_df_results)
    print(M_cords_x)
    print(len(M_cords_x))

    # Salva tabella
    s = M_df_results.style
    s.clear()
    s.caption = LOAD_TYPE
    s.format('{:.2f}', subset=Table.make_header(len(trave.spans)))
    with open(EXPORT_FOLDER+LOAD_TYPE+"_M.tex", 'w') as file:
        file.write(s.to_latex(position='H', hrules=True, siunitx=True))

# TAGLIO ------------------------

    V_cords_x = V.s_func
    V_cords_y_pos, V_cords_y_neg = V.inviluppo()
#header=Table.make_header(len(trave.spans)),
    V_df_results = Table.create_dataframe(        
        header = Table.make_header(len(trave.spans)),
        rows = Table.make_body(V_cords_x, V_cords_y_pos, V_cords_y_neg,trave.spans_cum_lenght()),
        index = ["s", "V_pos","V_neg"]
    )  
    print(V_df_results)
    print(V_cords_x)
    print(len(V_cords_x))

    # Salva tabella
    s = V_df_results.style
    s.clear()
    s.caption = LOAD_TYPE
    s.format('{:.2f}', subset=Table.make_header(len(trave.spans)))
    with open(EXPORT_FOLDER+LOAD_TYPE+"_V.tex", 'w') as file:
        file.write(s.to_latex(position='H', hrules=True, siunitx=True))

if __name__ == "__main__":
    run(trave)