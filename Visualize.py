import matplotlib.pyplot as plt
import numpy as np

from cycler import cycler
from matplotlib.collections import EventCollection
from Mapper import Mapper, BLEURTMapper_1
from typing import Tuple



class Visualize:

    def __init__(self):
        pass

    def visualize(self,\
            map_to_y : Mapper,\
            results : list,\
            x_data : np.ndarray,\
            color_arr : list = ['r', 'g', 'b', 'y'],\
            linestyle_arr : list = ['-', '--', ':', '-.']) -> None:
        
        mapper : Mapper = map_to_y(results)
        y_data : np.ndarray = mapper.mapping()
        shape : tuple = np.shape(y_data)
        scores : int = 1 if len(shape) == 1 else shape[1]

        default_cycler : cycler = (cycler(color=color_arr) +
            cycler(linestyle=linestyle_arr))
        plt.rc('axes', prop_cycle=default_cycler)
        fig : plt.figure = plt.figure()
        ax : plt.axes._subplots.AxesSubplot = fig.add_subplot(1, 1, 1)
        ax.set_ylim((-0.1,1))

        style_properties : list = list(default_cycler)

        print(x_data)
        print(y_data)

        for i in range(scores):
            ax.scatter(x_data, y_data[:, i])
            ax.plot(x_data, y_data[:, i])

            xevent : EventCollection = EventCollection(x_data,\
                color=style_properties[i]['color'],\
                linestyle=style_properties[i]['linestyle'],\
                linelength=0.025)
            yevent : EventCollection = EventCollection(y_data[:, i],\
                color=style_properties[i]['color'],\
                linestyle=style_properties[i]['linestyle'],\
                linelength=0.0025,\
                orientation='vertical')

            ax.add_collection(xevent)
            ax.add_collection(yevent)

        ax.set_title("Metric performance over increasing damaged data\n" + mapper.description())
        ax.set_ylabel('Scores')
        ax.set_xlabel('Percentage of perturbation')
        plt.gca().legend(tuple(mapper.key_names))
        plt.show()
        # TODO add description



if __name__ == "__main__":
    test : list = [
        {'P': [0.7943, 0.7597, 0.8081], 'R': [0.7889, 0.9368, 0.8825], 'F1': [0.7916, 0.8390, 0.8436],},
        {'P': [0.5, 0.7, 0.81], 'R': [0.9, 0.3, 0.25], 'F1': [0.4, 0.03, 0.36]},
        {'P': [0.4123, 0.592, 0.5132], 'R': [0.712, 0.212, 0.312], 'F1': [0.5123, 0.2, 0.46]}]
    #y = np.random.rand(10, 3)
    x = [0.0, 0.1, 0.2]

    Visualize().visualize(BLEURTMapper_1, test, x, color_arr=['c', 'm', 'y', 'k'])
