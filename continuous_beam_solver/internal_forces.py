from continuous_beam_solver.span_beam import Beam
from continuous_beam_solver.plotting import Plot
from continuous_beam_solver.global_variables import *

import numpy as np
import sympy as sp


def transpose_x(x:list, list_of_points:list, delta:float) -> list:
    """
    list_of_points is a list with cum lenght and x where is the max values. Use Table.makebody[0] to calculate it
    """
    s_tras_neg = x.copy()
    s_tras_pos = x.copy()

    for i in range(2,len(list_of_points),2):
        for j in range(len(s_tras_neg)):
            if s_tras_neg[j] > list_of_points[i-2] and s_tras_neg[j] < list_of_points[i]:
                if s_tras_neg[j] < list_of_points[i-1]:
                    s_tras_neg[j] = s_tras_neg[j] + delta
                else:
                    s_tras_neg[j] = s_tras_neg[j] - delta

    for i in range(2,len(list_of_points),2):
        for j in range(len(s_tras_pos)):
            if s_tras_neg[j] > list_of_points[i-2] and s_tras_pos[j] < list_of_points[i]:
                if s_tras_pos[j] < list_of_points[i-1]:
                    s_tras_pos[j] = s_tras_pos[j] - delta
                else:
                    s_tras_pos[j] = s_tras_pos[j] + delta
            
    return s_tras_pos, s_tras_neg


class InternalForce:
    def __init__(self, beam:Beam, x:list[sp.Matrix], r:list[sp.Matrix]):
        """x and r are the solutions taken from Solve.generate_expanded_x_solutions() 
        and Solve.generate_R_solutions(x)"""

        self.beam = beam
        self.nCampate =  len(beam.spans)
        self.x = x
        self.r = r
        self.s_func = np.arange(0, beam.spans_total_lenght(), ARANGE_STEP) # points on X axe
        #self.s_func = np.linspace(0, beam.spans_total_lenght(), num=1000)

    def calculate_internal_force_span_Q_1(self, span_Q : int): #TODO _1 _func
        pass

    def internal_force_beam_Q_1(self) -> list: #TODO  _1 _func + dividere 
        """"
        Return Y values of bending moment or shear with unitary Q load for the entire beam. It takes the values from bending_moment_span_Q
        """
        nCampate =  self.nCampate
        total_lenght = self.beam.spans_total_lenght()

        # substituting s_func points into the lambdify function -> list of Y points
        m_tot_1 = np.sum([self.calculate_internal_force_span_Q_1(span_Q = span)(self.s_func) for span in range(nCampate)], axis=0)
        return m_tot_1 #TODO m_beam_Q_1
    
    #def inviluppo_plot()

    def internal_force_beam_Q_real_values(self, combination):
        """
        Return Y values of bending moment or shear with the values of Q load substituted for each span
        """
        nCampate =  self.nCampate
        total_lenght = self.beam.spans_total_lenght()
        #Q_list = self.combinations()[1]
        m_tot_Q_values = np.sum([combination[span] * self.calculate_internal_force_span_Q_1(span_Q = span)(self.s_func) for span in range(nCampate)], axis=0)

        return m_tot_Q_values

    def inviluppo(self): #TODO nome inviluppo
        """For each combination substitute the real values of Q and then create the y+ and y- of the inviluppo"""
        #total_lenght = self.beam.spans_total_lenght()

        combinations = self.beam.combinations()
        # TODO da capire se ha senso o no togliere questa parte di grafico:
        # the param 'initial=0' is used for: 
        # in np.max:   if y[i] < 0 then y[i] = 0 else y[i] = y[i]
        # in np.min:   if y[i] > 0 then y[i] = 0 else y[i] = y[i]
        inviluppo_pos = np.max([self.internal_force_beam_Q_real_values(combinations[comb]) for comb in range(len(combinations))], axis=0, initial=0)
        inviluppo_neg = np.min([self.internal_force_beam_Q_real_values(combinations[comb]) for comb in range(len(combinations))], axis=0, initial=0)

        #inviluppo_pos = [inviluppo_max[i] if inviluppo_max[i] >  inviluppo_min[i] else inviluppo_min[i] for i in range(len(inviluppo_max))]
        #inviluppo_neg = [inviluppo_max[i] if inviluppo_max[i] <  inviluppo_min[i] else inviluppo_min[i] for i in range(len(inviluppo_max))]
        #inviluppo_pos = [0. if inviluppo_max[i] < 0  else inviluppo_max[i] for i in range(len(inviluppo_max))]

        return inviluppo_pos, inviluppo_neg
    
    def plot_inviluppo(self):
        """Plot the inviluppo() using Plot.my_plot_style"""
        fig, ax = Plot.plot_list_of_y_points(
                    s_func = self.s_func, 
                    list_of_y_points = self.inviluppo(), 
                    title =r"Inviluppo",
                    x_thicks = self.beam.spans_cum_lenght(), #aggiungere qui gli altri punti
                    y_label = r"M", 
                    color = PLOTTING_COLOR )
        return  fig, ax
    
    def plot_inviluppo_trasposed(self, list_of_points:list, delta:float): #TODO nome trasposed
        """Plot the inviluppo() using Plot.my_plot_style"""
        s_tras_pos, s_tras_neg = transpose_x(
            x=self.s_func, 
            list_of_points=list_of_points, 
            delta=delta)
        fig, ax = Plot.plot_list_of_y_points_transposed(
                    s_tras_pos = s_tras_pos, 
                    s_tras_neg = s_tras_neg,
                    list_of_y_points = self.inviluppo(), 
                    title =r"Inviluppo trasposto",
                    x_thicks = self.beam.spans_cum_lenght(), #aggiungere qui gli altri punti
                    y_label = r"M", 
                    color = PLOTTING_COLOR )
        return  fig, ax

    def plot_beam_Q_1(self):
        """Plot the internal_force_beam_Q_1() using Plot.my_plot_style"""
        fig, ax = Plot.plot_y_points(
                    s_func = self.s_func, 
                    y_points = self.internal_force_beam_Q_1(), 
                    title = "Carico unitario",
                    x_thicks = self.beam.spans_cum_lenght(), 
                    y_label = r"M", 
                    color = PLOTTING_COLOR )
        return  fig, ax
    
    def plot_span_Q_1(self, span_Q:int):
        """Plot the calculate_internal_force_span_Q_1(span_Q) using Plot.my_plot_style. The first span is 0"""
        fig, ax = Plot.plot_y_points(
                    s_func = self.s_func, 
                    y_points = self.calculate_internal_force_span_Q_1(span_Q)(self.s_func), 
                    title = f"Q = 1 only in span number {span_Q + 1}",
                    x_thicks = self.beam.spans_cum_lenght(), 
                    y_label = r"M", 
                    color = PLOTTING_COLOR )
        return  fig, ax

class BendingMoment(InternalForce):
    def __init__(self,beam:Beam, x:list[sp.Matrix], r:list[sp.Matrix]):
        super().__init__(beam, x, r)

    def calculate_internal_force_span_Q_1(self, span_Q : int) -> sp.lambdify : #TODO _1 _func
        """
        Compute the bending moment lamdified function for a "span_Q", 
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
        #s_lambdify  = np.linspace(0, total_lenght, 1000)
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

class Shear(InternalForce):
    def __init__(self,beam:Beam, x:list[sp.Matrix], r:list[sp.Matrix]):
        super().__init__(beam, x, r)

    def calculate_internal_force_span_Q_1(self, span_Q : int) -> sp.lambdify : 
        """
        Compute the shear lamdified function for a "span_Q", 
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
        v_i = [
                    (r[span_Q][n_span]  - (I[span_Q,n_span]*(s-cum_lenghts[span_Q]))) \
                    * (sp.Heaviside(s-cum_lenghts[n_span]) - sp.Heaviside(s-cum_lenghts[n_span+1])) \
                for n_span in range(nCampate)
            ]
        v_i_lambdify = sp.lambdify(s,np.sum(v_i,axis=0))
    # ---- With numpy: 
        # Not tried. See the BendingMoment Class instead

        return v_i_lambdify #TODO m_span_Q_1

    def plot_inviluppo(self):
        """Plot the inviluppo() using Plot.my_plot_style"""
        fig, ax = Plot.plot_list_of_y_points(
                    s_func = self.s_func, 
                    list_of_y_points = self.inviluppo(), 
                    title =r"Inviluppo",
                    x_thicks = self.beam.spans_cum_lenght(), #aggiungere qui gli altri punti
                    y_label = r"V", 
                    color = PLOTTING_COLOR )
        ax.invert_yaxis()
        return  fig, ax
