import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams['text.usetex'] = True
class Plot:
    def my_plot_style(list_of_y_points:list[list] or list, title, x_thicks:list, y_label:str):
        """This is just a style definitions for plot methods. 
        list_of_y_points can be an array of y points or a list of arrays of y points (like in the inviluppo methods)
        """
        fig, ax = plt.subplots(1,1, figsize = (10, 5), tight_layout=True)
        ax.invert_yaxis()
        ax.set_xlim(x_thicks[0], x_thicks[-1])
        ax.set_ylim(np.max(list_of_y_points), np.min(list_of_y_points) ) # invertiti perché l'asse è invertito
        ax.grid("True")
        ax.axhline(0, color='grey', linewidth=2)
        #ax.vlines(cum_lenghts, ymin=y_min, ymax=y_max,linestyles='dotted')
        #ax.set_xticklabels(cum_lenghts) # per il nome campata magari
        ax.set_xticks(x_thicks) #TODO aggiungere xthick nel massimo in mezzeria
        #ax.set_yticks(np.arange(-300, 250, step=50)) #TODO
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(r"L")
        ax.set_ylabel(y_label) #TODO aggiungere l'if se si usa per il taglio
        return  fig, ax

    def plot_y_points(s_func:list, y_points:list, title:str, x_thicks:list, y_label:str, color:str):
        """Plot the bending_moment_beam_Q_1() using my_plot_style"""
        #y_points = self.bending_moment_beam_Q_1()
        fig, ax = Plot.my_plot_style(y_points, title, x_thicks, y_label)

        ax.fill_between(s_func, y_points , linewidth=0, color=color)

        return  fig, ax
    
    def plot_list_of_y_points(s_func:list, list_of_y_points:list[list], title:str, x_thicks:list, y_label:str, color:str):
        """Plot the inviluppo() using my_plot_style"""
        #list_of_y_point = self.inviluppo()
        fig, ax = Plot.my_plot_style(list_of_y_points, title, x_thicks, y_label)
        
        for y_plot in list_of_y_points:
            ax.fill_between(s_func, y_plot , linewidth=0, color=color)
        
        return  fig, ax

