from VisData import VisData

import numpy as np
import matplotlib.pyplot as plt


from cycler import cycler
from matplotlib.collections import EventCollection


class Visualize_:


    def __init__(self,\
            color_arr : list = ['c', 'm', 'y', 'k'],\
            linestyle_arr : list = ['-', '--', ':', '-.']):

        self.color_arr : list = color_arr
        self.linestyle_arr : list = linestyle_arr
        self.fig : plt.figure = plt.figure()
        self.fig.suptitle("Metric performance over increasingly damaged data")
        self.default_cycler : cycler = (cycler(color=self.color_arr) +
            cycler(linestyle=self.linestyle_arr))



    def add_subplot(self, data : list, ylim : tuple = (-0.1,1.1)) -> None:

        if type(data) != "VisData":
            raise Exception("ERROR: data must be a list of VisData classes!")
        
        scores : int = 0
        if False in map(lambda x : x.scores == data[0].scores, data):
            raise Exception("ERROR: data must have the same amount of scores")
        else:
            scores = data[0].scores

        # ax : plt.axes._subplots.AxesSubplot = self.fig.add_subplot(math.ceil((index + 1) / self.max_len_vis), self.max_len_vis, (index % self.max_len_vis) + 1)
        ax : plt.axes._subplots.AxesSubplot = self.fig.add_subplot(2,2, 1)
        ax.set_xlim((-0.1,1))
        ax.set_ylim(ylim)

        style_properties : list = list(self.default_cycler)

        x_data : np.ndarray = np.asarray([e.degree for e in data])
        y_data : np.ndarray = np.asarray([e.scatter] for e in data)

        for i in range(scores):
            ax.scatter(x_data, y_data[:, i])

        
    def draw_plot(self):
        plt.show()