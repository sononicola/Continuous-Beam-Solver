from continuous_beam_solver.span_beam import Beam
from continuous_beam_solver.plotting import Plot
from continuous_beam_solver.global_variables import *

import numpy as np
import sympy as sp

class BendingMoment:
    def __init__(self, beam:Beam, x:list[sp.Matrix], r:list[sp.Matrix]):
        """x and r have solutions taken from Solve.generate_expanded_x_solutions() 
        and Solve.generate_R_solutions(x)"""

        self.beam = beam
        self.nCampate =  len(beam.spans)
        self.x = x
        self.r = r
        self.s_func = np.arange(0, beam.spans_total_lenght(), ARANGE_STEP) # points on X axe
        #self.s_func = np.linspace(0, beam.spans_total_lenght(), num=1000)

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
                    title =r"Inviluppo",
                    x_thicks = self.beam.spans_cum_lenght(), #aggiungere qui gli altri punti
                    y_label = r"M", 
                    color = "r" )
        return  fig

    def plot_beam_Q_1(self):
        """Plot the bending_moment_beam_Q_1() using Plot.my_plot_style"""
        fig, ax = Plot.plot_y_points(
                    s_func = self.s_func, 
                    y_points = self.bending_moment_beam_Q_1(), 
                    title = "Carico unitario",
                    x_thicks = self.beam.spans_cum_lenght(), 
                    y_label = r"M", 
                    color = "r" )
        return  fig
    
    def plot_span_Q_1(self, span_Q:int):
        """Plot the bending_moment_span_Q_1(span_Q) using Plot.my_plot_style. The first span is 0"""
        fig, ax = Plot.plot_y_points(
                    s_func = self.s_func, 
                    y_points = self.bending_moment_span_Q_1(span_Q)(self.s_func), 
                    title = f"Q = 1 only in span number {span_Q + 1}",
                    x_thicks = self.beam.spans_cum_lenght(), 
                    y_label = r"M", 
                    color = "r" )
        return  fig



