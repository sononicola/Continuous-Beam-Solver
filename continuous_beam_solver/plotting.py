import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def my_plot_style(list_of_y_points:list[list] or list, title, x_thicks:list, y_label:str):
        """This is just a style definitions for plot methods. 
        list_of_y_points can be an array of y points or a list of arrays of y points (like in the inviluppo methods)
        """
        fig, ax = plt.subplots(1,1, figsize = (12, 6), tight_layout=True)
       
        ax.invert_yaxis()
        ax.set_xlim(x_thicks[0], x_thicks[-1])
        ax.set_ylim(np.max(list_of_y_points), np.min(list_of_y_points) ) # invertiti perché l'asse è invertito
        ax.grid("True")
        ax.axhline(0)
        #ax.vlines(cum_lenghts, ymin=y_min, ymax=y_max,linestyles='dotted')
        #ax.set_xticklabels(cum_lenghts) # per il nome campata magari
        ax.set_xticks(np.round(x_thicks,2)) #TODO aggiungere xthick nel massimo in mezzeria
        #ax.set_yticks(np.arange(-300, 250, step=50)) #TODO 
        ax.tick_params(axis='both', labelsize=12)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("L", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14) #TODO aggiungere l'if se si usa per il taglio
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

    def plot_list_of_y_points_transposed(s_tras_pos:list, s_tras_neg:list, list_of_y_points:list[list], title:str, x_thicks:list, y_label:str, color:str):
        """Plot the inviluppo() using my_plot_style"""
        #list_of_y_point = self.inviluppo()
        fig, ax = Plot.my_plot_style(list_of_y_points, title, x_thicks, y_label)
        
        ax.fill_between(s_tras_pos, list_of_y_points[0], linewidth=0, color=color)
        ax.fill_between(s_tras_neg, list_of_y_points[1] , linewidth=0, color=color)
        
        return  fig, ax

