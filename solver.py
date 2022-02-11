import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class Span:
    def __init__(self, lenght: float, ej: float, q_max: float = 0., q_min: float = 0.):
        self.lenght = lenght
        self.ej = ej
        self.q_max = q_max
        self.q_min = q_min

    def set_lenght(self):  # boh non servirà credo
        pass

class Beam: 
    def __init__(self, spans: list[Span] , supports: str = "no-incastre"):
        self.spans = spans
        self.supports = supports # suddividere in  left and right

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

class Compute:
    def __init__(self,beam: Beam):
        self.beam = beam
        self.nCampate =  len(beam.spans)
        
    def generate_simbolic_variables(self):
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
        supports = self.beam.supports
        flex_gen  = self.generate_Flex_matrix()
        P_gen = self.generate_P_vector_Q()

        if supports == "no-incastre":
            flex_rid = flex_gen[1:nCampate,1:nCampate]
            P_rid    = P_gen[1:nCampate]
        elif supports == "incastre-left":   
            flex_rid =  flex_gen[0:nCampate,0:nCampate]
            P_rid    = P_gen[0:nCampate]
        elif supports == "incastre-right":
            flex_rid = flex_gen[1:nCampate+1,1:nCampate+1]
            P_rid    = P_gen[1:nCampate+1]
        elif supports == "double-incastre":
            flex_rid = sp.Matrix.copy(flex_gen)
            P_rid    = sp.Matrix.copy(P_gen)

        flex_rid = sp.Matrix(flex_rid)
        P_rid = sp.Matrix(P_rid)
        return flex_rid, P_rid

    def generate_reduced_x_solutions(self) -> list:
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
    def generate_expanded_x_solutions(self) -> list:
        """
        In base of boundary conditions return to initial lenghts the x_solution_vectors calculated in generate_reduced_x_solutions(self)
        """
        nCampate =  self.nCampate
        supports = self.beam.supports
        list_of_reduced_x_solution_vectors = self.generate_reduced_x_solutions()

    # init identic of the reduced list and then add or not the zeros
        list_of_expanded_x_solution_vectors = list_of_reduced_x_solution_vectors
        for n_span in range(nCampate):
            if supports == "no-incastre": # 0 prima e dopo -- damodificare

                list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(sp.zeros(1,1), list_of_reduced_x_solution_vectors[n_span])
            elif supports == "incastre-left":   # 0 dopo
                pass
            elif supports == "incastre-right": # 0 prima
                list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(sp.zeros(1,1), list_of_reduced_x_solution_vectors[n_span])
            elif supports == "double-incastre": # no 0
                pass

        return list_of_expanded_x_solution_vectors

    def generate_R_solutions(self) -> list:
        """
        R
        """
        nCampate =  self.nCampate
        x = self.generate_expanded_x_solutions()
        lenghts = self.beam.spans_lenght() 

        list_of_R = [] # ---- oppure mettere - X + L/2: 
        for n_span in range(nCampate):
            mat1 = sp.Matrix([  (x[n_span][i+1] - x[n_span][i])/lenghts[i] for i in range(0,nCampate)  ])
            mat2 = sp.Matrix(lenghts[n_span]/2 * np.identity(nCampate)[n_span])
            list_of_R.append(mat1 + mat2)
        return list_of_R
          
    def bending_moment_span_Q(self, span_Q : int):
        """
        span_Q is the span where the distribuited load Q is applied
        """
        # TODO togliere i commentati
        nCampate =  self.nCampate
        #lenghts = self.beam.spans_lenght() 
        cum_lenghts = self.beam.spans_cum_lenght()
        total_lenght = self.beam.spans_total_lenght()

        x = self.generate_expanded_x_solutions() # List of matrixes
        r = self.generate_R_solutions() # List of matrixes
     
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
        m_i_lambdify = sp.lambdify(s,-np.sum(m_i,axis=0))
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

        return m_i_lambdify

    def bending_moment_beam_Q(self): #TODO  
        nCampate =  self.nCampate
        return [self.bending_moment_span_Q(span_Q) for span_Q in range(nCampate)]

    def plot_bending_moment_beam_Q(self):
        nCampate =  self.nCampate 
        total_lenght = self.beam.spans_total_lenght()

        s_func = np.arange(0, total_lenght, .001)
        m_tot_un = np.sum([self.bending_moment_span_Q(span_Q = span)(s_func) for span in range(nCampate)], axis=0)

        fig = plt.figure(figsize = (10, 5))
        plt.plot(s_func,m_tot_un)
        plt.axhline(0, color='black')
        plt.show()
        
        return fig
J = (0.3 * 0.5**3)/12 # m4
EJ =  31476*1000000*J/1000 # Mpa * m4 -> N*m2 -> kN*m2

c_1 = Span(lenght = 3.00, ej = EJ)
c_2 = Span(lenght = 4.50, ej = EJ)
c_3 = Span(lenght = 4.00, ej = EJ)
c_4 = Span(lenght = 5.00, ej = EJ)
c_5 = Span(lenght = 6.15, ej = EJ)
c_6 = Span(lenght = 4.00, ej = EJ)

trave = Beam(spans = [c_1, c_2, c_3, c_4, c_5, c_6], supports='incastre-right')

run = Compute(trave)
print(f"{trave.spans_lenght() = }")
print(f"{trave.spans_total_lenght() = }")
print(f"{trave.spans_cum_lenght() = }")
print(f"{trave.spans_ej() = }")
print(f"{trave.spans_q_max() = }")
print(f"{trave.spans_q_min() = }")

print("x",run.generate_expanded_x_solutions())
print("R",run.generate_R_solutions())

#print(run.bending_moment_span_Q(0))
#plt.plot(run.bending_moment_span_Q(0))

#s_func = np.arange(0, 26.5, .001)
#for i in range(6):
 #   m = run.bending_moment_span_Q(span_Q=i)(s_func)
 #   plt.plot(s_func,m)
 #   plt.show()
 #   plt.close()

run.plot_bending_moment_beam_Q()
