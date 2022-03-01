import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from plotting import Plot
from utils import *

from tables import Table
class Span:
    def __init__(self, lenght: float, ej: float, q_max: float = 0., q_min: float = 0.):
        self.lenght = lenght
        self.ej = ej
        self.q_max = q_max
        self.q_min = q_min

    def set_lenght(self):  # boh non servirà credo
        pass

class Beam: 
    def __init__(self, spans: list[Span], left_support: str, right_support: str):
        """
        Avaiable left and right supports: "Simple", "Fixed". "Free" not implemented yet!" 
        """
        self.spans = spans
        self.left_support = left_support
        self.right_support = right_support

    def get_spans(self): # boh non servirà credo
        return self.spans

    def add_span(self, new_span: Span):
        """Add a single Span object to the spans list. 
        Don't add a list of object like add_span([x1,x2]), but use add_list_of_spans([x1,x2]) instead!
        
        x1 = Span(...) \n
        beam = Beam([]) \n
        beam.add_span(x1)
        """
        self.spans.append(new_span)
    
    def add_list_of_spans(self, list_of_spans: list[Span]):
        """
        Add a list of object Span to the spans list. 
        To add a single Span object use add_span(x1) insted!
        
        x1 = Span(...) \n
        x2 = Span(...) \n
        beam = Beam([]) \n
        beam.add_span([x1,x2])
        """
        self.spans.extend(list_of_spans)

    def spans_lenght(self) -> list:
        """Return a list with spans' lenghts"""
        return [span.lenght for span in self.spans]

    def spans_total_lenght(self) -> float:
        """Return the sum of spans lenght"""
        return np.sum(self.spans_lenght(), dtype=float)
    
    def spans_cum_lenght(self) -> list:
        """Return the cumulative sum of spans where the first element is 0"""
        cum_sum  = np.cumsum(self.spans_lenght(), dtype=float)
        cum_sum = np.insert(cum_sum,0,0.0) # Add a 0 at the beginning
        return cum_sum

    def spans_ej(self) -> list:
        """Return a list with spans' ej"""
        return [span.ej for span in self.spans]
    
    def spans_q_max(self) -> list:
        """Return a list with spans' q_max"""
        return [span.q_max for span in self.spans]
    
    def spans_q_min(self) -> list:
        """Return a list with spans' q_min"""
        return [span.q_min for span in self.spans]

    def combinations(self) -> list[list] :
        q_max_list = self.spans_q_max()
        q_min_list = self.spans_q_min()
        nCampate =  len(self.spans)

        #for testing:
        #q_max_list = ["S1", "S2", "S3", "S4", "S5", "S6"]
        #q_min_list = ["F1", "F2", "F3", "F4", "F5", "F6"]
        #nCampate = 6        
        
        # S F S F S F ...
        comb_1 = [q_max_list[i] if i%2 == 0 else q_min_list[i] for i in range(nCampate)]
        # F S F S F S...
        comb_2 = [q_max_list[i] if i%2 == 1 else q_min_list[i] for i in range(nCampate)]
        # S S F S F S...
        comb_3 = [q_max_list[0]]
        comb_3.extend([q_max_list[i] if i%2 == 1 else q_min_list[i] for i in range(1,nCampate)])

        comb_3 = [q_max_list[i] if i%2 == 1 else q_min_list[i] for i in range(nCampate)]
        comb_3[0] = q_max_list[0]



        #print(comb_1)
        #print(comb_2)
        #print(comb_3)
        return [comb_1,comb_2]

    # --- REAL SOLVING METHODS: ---
    # Using sympy:  symbolic -> reduced with BC -> subsituted with numeric values ->  solved the system ->  expanded to initial lenghts row, columns

    def symbolic_matrix_lenghts():
        pass

    def symbolic_matrix_Q():
        pass

    def symbolic_matrix_P():
        pass

    def symbolic_matrix_EJ():
        pass

    # --- PRINTING/PLOTTING AND STORING VALUES ---
    def plot_M_unitario(): # nomi migliori per tuti
        pass
   
    def plot_V_unitario():
        pass

    def plot_M_totale():
        pass
   
    def plot_V_totale():
        pass

    def tabella():
        pass

    def latex_passaggi_intermedi():
        pass

    # ---
    # Linee orizzontali al momento aggiungendo un parametro a Beam con il valore di M

    # Disegno tikz

class Solver:
    def __init__(self,beam: Beam):
        self.beam = beam
        self.nCampate =  len(beam.spans)
        
    def generate_simbolic_variables(self): #TODO mettere altre condizioni alle variabili?
        nCampate =  self.nCampate
        L   = sp.Matrix(nCampate, 1, [sp.symbols(f"L_{i}",real="True",nonnegative="True") for i in range(1, nCampate+1)]   )
        P   = sp.Matrix(nCampate, 1, [sp.symbols(f"P_{i}") for i in range(1, nCampate+1)]   )
        Q   = sp.Matrix(nCampate, 1, [sp.symbols(f"Q_{i}",positive="True") for i in range(1, nCampate+1)]   )
        EJ  = sp.Matrix(nCampate, 1, [sp.symbols(f"EJ_{i}") for i in range(1, nCampate+1)]  )
        return L, P, Q, EJ

    def generate_Flex_matrix(self) -> sp.Matrix:
        L, P, Q, EJ = self.generate_simbolic_variables()
        nCampate =  self.nCampate

    # --- Principal Diagonal of the matrix ----
        flex_diag_sx = sp.Matrix([sp.Rational(1,3) * L[i] * EJ[i]**-1 for i in range(0,nCampate)])
        # Aggiunge uno zero all'inizio diventando n+1
        flex_diag_sx = sp.Matrix.vstack(sp.zeros(1,1), flex_diag_sx) 

        flex_diag_dx = sp.Matrix([sp.Rational(1,3) * L[i] * EJ[i]**-1 for i in range(0,nCampate)])
        # Aggiunge uno zero alla fine diventando n+1:
        flex_diag_dx = sp.Matrix.vstack(flex_diag_dx , sp.zeros(1,1)) 

        flex_diag_tot = flex_diag_sx + flex_diag_dx

    # ---- Lower Diagonal of the matrix ----
        flex_lowerdiag = sp.Rational(1,2) * sp.Matrix(flex_diag_sx[1:nCampate+1])

    # ---- Upper Diagonal of the matrix ----
        flex_upperdiag = sp.Rational(1,2) * sp.Matrix(flex_diag_dx[0:-1])

    # ---- Total Flex matrix ----
        flex_gen = sp.zeros(nCampate+1,nCampate+1)
        for i in range(nCampate+1): 
            flex_gen[i,i] = flex_diag_tot[i] 
        for i in range(nCampate):
            flex_gen[i,i+1] = flex_upperdiag[i]
        for i in range(nCampate):
            flex_gen[i+1,i] = flex_lowerdiag[i]

        return flex_gen

    def generate_P_vector_Q(self) -> sp.Matrix:
        L, P, Q, EJ = self.generate_simbolic_variables()
        nCampate =  self.nCampate

        P_sx = sp.Matrix([sp.Rational(1,24) * Q[i] * L[i]**3 * EJ[i]**-1 for i in range(0,nCampate)]) 
        # Aggiunge uno zero all'inizio diventando n+1
        P_sx = sp.Matrix.vstack(sp.zeros(1,1), P_sx) 

        P_dx = sp.Matrix([sp.Rational(1,24) * Q[i] * L[i]**3 * EJ[i]**-1 for i in range(0,nCampate)])
        # Aggiunge uno zero alla fine diventando n+1:
        P_dx = sp.Matrix.vstack(P_dx , sp.zeros(1,1)) 

        return P_sx  + P_dx

    def generate_reduced_Flex_matrix_and_P_vector(self):
        nCampate =  self.nCampate
        left_support = self.beam.left_support
        right_support = self.beam.right_support
        flex_gen  = self.generate_Flex_matrix()
        P_gen = self.generate_P_vector_Q()

        #TODO aggiungere il Free support
        if left_support == "Simple" and right_support == "Simple":
            flex_rid = flex_gen[1:nCampate,1:nCampate]
            P_rid    = P_gen[1:nCampate]
        elif left_support == "Fixed" and right_support == "Simple":   
            flex_rid =  flex_gen[0:nCampate,0:nCampate]
            P_rid    = P_gen[0:nCampate]
        elif left_support == "Simple" and right_support == "Fixed":
            flex_rid = flex_gen[1:nCampate+1,1:nCampate+1]
            P_rid    = P_gen[1:nCampate+1]
        elif left_support == "Fixed" and right_support == "Fixed":
            flex_rid = sp.Matrix.copy(flex_gen)
            P_rid    = sp.Matrix.copy(P_gen)

        flex_rid = sp.Matrix(flex_rid)
        P_rid = sp.Matrix(P_rid)
        return flex_rid, P_rid

    def generate_reduced_x_solutions(self) -> list[sp.Matrix]:
        nCampate =  self.nCampate
        L, P, Q, EJ = self.generate_simbolic_variables()
        flex_rid, P_rid = self.generate_reduced_Flex_matrix_and_P_vector()
        # List of numeric values taken from Beam 
        lenghts = self.beam.spans_lenght() 
        ej = self.beam.spans_ej()

        # To solve the system for a generic Q=1 we have to subtistute values and then solve the systen for every span.
        # With Q values we have to subsistute them one at time and the must be other zero. 
        #   Example with the first span: Q1 = 1, Q2 = 0 , Q3 = 0, ...
        #   Example with the second span: Q1 =0, Q2 = 1 , Q3 = 0, ...
        # To do this I'm using the Identidy matrix

        # Every x_solution_vector is added to a list 
        list_of_reduced_x_solution_vectors = []
        for n_span in range(nCampate):
            flex_sub = flex_rid \
                .subs(zip(L,lenghts)) \
                .subs(zip(EJ,ej)) \
                .subs(zip(Q,np.array(sp.Identity(nCampate))[n_span]))

            P_sub = P_rid \
                .subs(zip(L,lenghts)) \
                .subs(zip(EJ,ej)) \
                .subs( zip(Q,np.identity(nCampate)[n_span]))

            # solve the system: # --- maybe there is a more efificient way 
            x = - flex_sub.inv() * P_sub
            list_of_reduced_x_solution_vectors.append(x)
        return list_of_reduced_x_solution_vectors

    def generate_expanded_x_solutions(self) -> list[sp.Matrix]:
        """
        In base of boundary conditions return to initial lenghts the x_solution_vectors calculated in generate_reduced_x_solutions(self)
        """
        nCampate =  self.nCampate
        left_support = self.beam.left_support
        right_support = self.beam.right_support
        list_of_reduced_x_solution_vectors = self.generate_reduced_x_solutions()

    # init with an identic list of the reduced one, and then add or not the zeros
        list_of_expanded_x_solution_vectors = list_of_reduced_x_solution_vectors #TODO
        for n_span in range(nCampate):
            if left_support == "Simple" and right_support == "Simple": # 0 prima e dopo
                # if there is only one span and no-incastre: 
                # 'list_of_reduced_x_solution_vectors' fails to compute an adeguated matrix beacuse there aren't solutions of the system, then 
                # 'generate_expanded_x_solutions' and 'generate_R_solutions' give an index error. So with this if i'm overwriting directly this case
                if self.nCampate > 1:
                    list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(sp.zeros(1,1), list_of_reduced_x_solution_vectors[n_span])
                    list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(list_of_expanded_x_solution_vectors[n_span], sp.zeros(1,1))
                else: 
                    list_of_expanded_x_solution_vectors[n_span] = sp.zeros(2,1)

            elif left_support == "Fixed" and right_support == "Simple":   # 0 dopo
                list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(list_of_reduced_x_solution_vectors[n_span], sp.zeros(1,1))

            elif left_support == "Simple" and right_support == "Fixed": # 0 prima
                list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(sp.zeros(1,1), list_of_reduced_x_solution_vectors[n_span])

            elif left_support == "Fixed" and right_support == "Fixed": # no 0
                pass

        return list_of_expanded_x_solution_vectors

    def generate_R_solutions(self, x:list[sp.Matrix]) -> list[sp.Matrix]:
        """
        R
        """
        nCampate =  self.nCampate
        #x = self.generate_expanded_x_solutions()
        lenghts = self.beam.spans_lenght() 

        list_of_R = [] # ---- oppure mettere - X + L/2: 
        for n_span in range(nCampate):
            mat1 = sp.Matrix([  (x[n_span][i+1] - x[n_span][i])/lenghts[i] for i in range(0,nCampate)  ])
            mat2 = sp.Matrix(lenghts[n_span]/2 * np.identity(nCampate)[n_span])
            list_of_R.append(mat1 + mat2)
        return list_of_R

class BendingMoment:
    def __init__(self, beam:Beam, x:list[sp.Matrix], r:list[sp.Matrix]):
        """x and r have solutions taken from Solve.generate_expanded_x_solutions() 
        and Solve.generate_R_solutions(x)"""

        self.beam = beam
        self.nCampate =  len(beam.spans)
        self.x = x
        self.r = r
        self.s_func = np.arange(0, beam.spans_total_lenght(), .001) # points on X axe

    def bending_moment_span_Q_1(self, span_Q : int): #TODO _1 _func
        """
        Compute the bending_moment_Q lamdified function for a "span_Q", 
        which is the span where the distribuited load Q is applied and the others Q is zero
        """
        # TODO togliere i commentati
        nCampate =  self.nCampate
        #lenghts = self.beam.spans_lenght() 
        cum_lenghts = self.beam.spans_cum_lenght()
        total_lenght = self.beam.spans_total_lenght()

        x = self.x # List of matrixes
        r = self.r # List of matrixes
     
        I = np.identity(nCampate)
        #span_i = 1 # campata vera
        #n_span = 0
    # ---- With Sympy:
        s = sp.Symbol('s')
        m_i = [
                    ((x[span_Q][n_span] + r[span_Q][n_span] * (s-cum_lenghts[n_span])) - ((I[span_Q,n_span]*(s-cum_lenghts[span_Q])**2)/2)) \
                    * (sp.Heaviside(s-cum_lenghts[n_span]) - sp.Heaviside(s-cum_lenghts[n_span+1])) \
                for n_span in range(nCampate)
            ]
        m_i_lambdify = sp.lambdify(s,np.sum(m_i,axis=0))
        s_lambdify  = np.linspace(0, total_lenght, 1000)
    # ---- With numpy: TODO maybe. doesnt work as expected the heaviside func
        #s = np.linspace(0, total_lenght, 1000)
        #m_i = [
        #           ((x[span_Q][n_span] + r[span_Q][n_span] * (s-cum_lenghts[n_span])) - ((I[span_Q,n_span]*(s-cum_lenghts[span_Q])**2)/2)) \
        #            * (np.heaviside(s-cum_lenghts[n_span],0) - np.heaviside(s-cum_lenghts[n_span+1],0)) \
        #        for n_span in range(nCampate)
        #    ]
        
        # m_i is a list of list. We want to sum each list inside, not the total of everything: so we need "axis=0"
        # Example from numpy documentation: np.sum([[0, 1], [0, 5]], axis=0) >>> array([0, 6])
        #return np.sum(m_i,axis=0) 

        return m_i_lambdify #TODO m_span_Q_1

    def bending_moment_beam_Q_1(self) -> list: #TODO  _1 _func + dividere 
        """"
        Return Y values of bending moment with unitary Q load for the entire beam. It takes the values from bending_moment_span_Q
        """
        nCampate =  self.nCampate
        total_lenght = self.beam.spans_total_lenght()

        # substituting s_func points into the lambdify function -> list of Y points
        m_tot_1 = np.sum([self.bending_moment_span_Q_1(span_Q = span)(self.s_func) for span in range(nCampate)], axis=0)
        return m_tot_1 #TODO m_beam_Q_1
    
    #def inviluppo_plot()

    def bending_moment_beam_Q_real_values(self, combination):
        """
        Return Y values of bending moment with the valuesof Q load substituted for each span
        """
        nCampate =  self.nCampate
        total_lenght = self.beam.spans_total_lenght()
        #Q_list = self.combinations()[1]
        m_tot_Q_values = np.sum([combination[span] * self.bending_moment_span_Q_1(span_Q = span)(self.s_func) for span in range(nCampate)], axis=0)

        return m_tot_Q_values

    def inviluppo(self): #TODO nome inviluppo
        """For each combination substitute the real values of Q and then create the y+ and y- of the inviluppo"""
        #total_lenght = self.beam.spans_total_lenght()

        combinations = self.beam.combinations()
        # TODO da capire se ha senso o no togliere questa parte di grafico:
        # the param 'initial=0' is used for: 
        # in np.max:   if y[i] < 0 then y[i] = 0 else y[i] = y[i]
        # in np.min:   if y[i] > 0 then y[i] = 0 else y[i] = y[i]
        inviluppo_pos = np.max([self.bending_moment_beam_Q_real_values(combinations[comb]) for comb in range(len(combinations))], axis=0, initial=0)
        inviluppo_neg = np.min([self.bending_moment_beam_Q_real_values(combinations[comb]) for comb in range(len(combinations))], axis=0, initial=0)

        #inviluppo_pos = [inviluppo_max[i] if inviluppo_max[i] >  inviluppo_min[i] else inviluppo_min[i] for i in range(len(inviluppo_max))]
        #inviluppo_neg = [inviluppo_max[i] if inviluppo_max[i] <  inviluppo_min[i] else inviluppo_min[i] for i in range(len(inviluppo_max))]
        #inviluppo_pos = [0. if inviluppo_max[i] < 0  else inviluppo_max[i] for i in range(len(inviluppo_max))]

        return inviluppo_pos, inviluppo_neg
    
    def plot_inviluppo(self):
        """Plot the inviluppo() using Plot.my_plot_style"""
        fig, ax = Plot.plot_list_of_y_points(
                    s_func = self.s_func, 
                    list_of_y_points = self.inviluppo(), 
                    title =r"\bf{Inviluppo}",
                    x_thicks = self.beam.spans_cum_lenght(), #aggiungere qui gli altri punti
                    y_label = r"$M$", 
                    color = "r" )
        plt.show()
        plt.close()
        return  fig

    def plot_beam_Q_1(self):
        """Plot the bending_moment_beam_Q_1() using Plot.my_plot_style"""
        fig, ax = Plot.plot_y_points(
                    s_func = self.s_func, 
                    y_points = self.bending_moment_beam_Q_1(), 
                    title = "Carico unitario",
                    x_thicks = self.beam.spans_cum_lenght(), 
                    y_label = r"$M$", 
                    color = "r" )
        plt.show()
        plt.close()
        return  fig
    
    def plot_span_Q_1(self, span_Q:int):
        """Plot the bending_moment_span_Q_1(span_Q) using Plot.my_plot_style. The first span is 0"""
        fig, ax = Plot.plot_y_points(
                    s_func = self.s_func, 
                    y_points = self.bending_moment_span_Q_1(span_Q)(self.s_func), 
                    title = f"Q = 1 only in span number {span_Q + 1}",
                    x_thicks = self.beam.spans_cum_lenght(), 
                    y_label = r"$M$", 
                    color = "r" )
        plt.show()
        plt.close()
        return  fig



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



def testing(beam: Beam):
    print(f"{beam.spans_lenght() = }")
    print(f"{beam.spans_total_lenght() = }")
    print(f"{beam.spans_q_min() = }")
    print(f"{beam.spans_cum_lenght() = }")
    print(f"{beam.spans_ej() = }")
    print(f"{beam.spans_q_max() = }")
    print(f"{beam.combinations() = }")
#testing(trave)
#quit()

def run(beam: Beam):
    sol = Solver(beam)
    x = sol.generate_expanded_x_solutions()
    #print(f"x = {x}")
    r = sol.generate_R_solutions(x)
    #print(f"R = {r}")

    M = BendingMoment(beam, x, r)
    
    #M.plot_span_Q_1(0)
    #M.plot_beam_Q_1()
    #M.plot_inviluppo()

    print(find_local_max_xy(M.s_func, M.inviluppo()[0], 5., 26., tol=0.001/2))
    
    print(np.where((M.s_func == trave.spans_cum_lenght()[6])))
    print(np.where((M.s_func >= trave.spans_cum_lenght()[6] - 0.001/2) & (M.s_func <= trave.spans_cum_lenght()[6] + 0.001/2)))
    

    #fig, ax = plt.subplots(1,1, figsize = (10, 5), tight_layout=True)
    #ax.invert_yaxis()
    #ax.fill_between(M.s_func, M.inviluppo()[0], color='r')
    #ax.fill_between(M.s_func, M.inviluppo()[1], color='b')
    #ax.axhline(0, color='grey', linewidth=2)
    
    #plt.show()

    #x_row = Table.make_principal_row(x = M.s_func, y = M.inviluppo()[0],  cum_lenghts = trave.spans_cum_lenght())
    #print(x_row)


    print(find_local_max_xy(M.s_func, M.inviluppo()[0], trave.spans_cum_lenght()[0], trave.spans_cum_lenght()[0], tol=0.001/2))
    maxs = Table.list_of_max_indexes(x = M.s_func, y = M.inviluppo()[0], list_of_points = trave.spans_cum_lenght(), tol=0.001/2)
    print(maxs)
    print(Table.list_of_min_indexes(x = M.s_func, y = M.inviluppo()[1], list_of_points = trave.spans_cum_lenght(), tol=0.001/2))
    print([M.s_func[i] for i in maxs[:-1]])

run(trave)
