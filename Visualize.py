from __future__ import annotations

import matplotlib.pyplot as plt
import math
import numpy as np

from cycler import cycler
from matplotlib.collections import EventCollection
from Mapper import Mapper, AverageMapper
from typing import Tuple



class Visualize:
    """
    TODO
    """

    def __init__(self,\
            results : list,\
            metrics : list = ["ME", "BLEURT", "BERTScore"],\
            color_arr : list = ['c', 'm', 'y', 'k'],\
            linestyle_arr : list = ['-', '--', ':', '-.']) -> None:

        if len(set(metrics)) != len(metrics):
            raise Exception("ERROR: Each metric can only be used onces!") 

        self.results : list = results
        self.color_arr : list = color_arr
        self.linestyle_arr : list = linestyle_arr
        self.metrics : np.ndarray = np.asarray(metrics)
        self.max_len_vis : int = math.ceil(len(metrics) / 2)

        self.default_cycler : cycler = (cycler(color=self.color_arr) +
        cycler(linestyle=self.linestyle_arr))
        plt.rc('axes', prop_cycle=self.default_cycler)
        self.fig : plt.figure = plt.figure()
        self.fig.suptitle("Metric performance over increasing damaged data")



    def draw_subplot(self,\
            map_to_y : Mapper,\
            metric : str,\
            ylim : tuple = (-0.1,1.1)) -> None:
        """
        TODO
        """
        
        # Prepare data for specific metric
        x, y = list(zip(*list(map(lambda elem : (elem[0], elem[1][metric]), self.results))))
        x_data, y = list(x), list(y)
        index : int = np.where(self.metrics == metric)[0][0]

        mapper : Mapper = map_to_y(y)
        y_data : np.ndarray = mapper.mapping()
        shape : tuple = np.shape(y_data)
        scores : int = 1 if len(shape) == 1 else shape[1]

        # ax : plt.axes._subplots.AxesSubplot = self.fig.add_subplot(math.ceil((index + 1) / self.max_len_vis), self.max_len_vis, (index % self.max_len_vis) + 1)
        ax : plt.axes._subplots.AxesSubplot = self.fig.add_subplot(2,2, index + 1)
        ax.set_xlim((-0.1,1))
        ax.set_ylim(ylim)

        style_properties : list = list(self.default_cycler)

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
                linelength=0.025,\
                orientation='vertical')

            ax.add_collection(xevent)
            ax.add_collection(yevent)

        ax.set_title((f"%s:" % metric)+ mapper.description())
        ax.set_ylabel('Scores')
        ax.set_xlabel('Percentage of perturbation')
        ax.legend(tuple(mapper.key_names))

    
    def visualize(self) -> Visualize:

        if "ME" in self.metrics:
            self.draw_subplot(Mapper, "ME")    

        if "BERTScore" in self.metrics:
            self.draw_subplot(AverageMapper, "BERTScore")
        
        if "BLEURT" in self.metrics:
            self.draw_subplot(AverageMapper, "BLEURT", ylim=(-2,2))

        if "GRUEN" in self.metrics:
            self.draw_subplot(AverageMapper, "GRUEN")
        
        plt.show()




if __name__ == "__main__":
    test_ : list = [(0.0, {
        'ME': {'Peterson': 1.0, 'Schnabel': 1.0, 'CAPTURE': 1.0},
        'BERTScore': {'P': [0.7943, 0.8081], 'R': [0.7889, 0.8825], 'F1': [0.7916, 0.8436]},
        'BLEURT': {'BLEURT': [-1.251094102859497, -1.0400080680847168]},
        'GRUEN': {'GRUEN': [0.7391475883349699, 0.4926017081756097]},
        'NUBIA': [{'nubia_score': 0.05430618350698457, 'features': {'semantic_relation': 1.5312963724136353, 'contradiction': 69.75854635238647, 'irrelevancy': 12.858390808105469, 'logical_agreement': 17.383064329624176, 'grammar_ref': 5.018170356750488, 'grammar_hyp': 8.034308433532715}}, {'nubia_score': 0.15190720827875148, 'features': {'semantic_relation': 0.8979882001876831, 'contradiction': 0.061019702116027474, 'irrelevancy': 2.988027408719063, 'logical_agreement': 96.95096015930176, 'grammar_ref': 7.9379754066467285, 'grammar_hyp': 6.1522674560546875}}]
        }),
        (0.1, {
            'ME': {'Peterson': 0.75, 'Schnabel': 0.9, 'CAPTURE': 1.0},
            'BERTScore': {'P': [0.7943, 0.8081], 'R': [0.7889, 0.8825], 'F1': [0.7916, 0.8436]},
            'BLEURT': {'BLEURT': [-0.9123, -0.8400080680847168]},
            'GRUEN': {'GRUEN': [0.52, 0.23]},
            'NUBIA': [{'nubia_score': 0.05430618350698457, 'features': {'semantic_relation': 1.5312963724136353, 'contradiction': 69.75854635238647, 'irrelevancy': 12.858390808105469, 'logical_agreement': 17.383064329624176, 'grammar_ref': 5.018170356750488, 'grammar_hyp': 8.034308433532715}}, {'nubia_score': 0.15190720827875148, 'features': {'semantic_relation': 0.8979882001876831, 'contradiction': 0.061019702116027474, 'irrelevancy': 2.988027408719063, 'logical_agreement': 96.95096015930176, 'grammar_ref': 7.9379754066467285, 'grammar_hyp': 6.1522674560546875}}]
        })]
    test : list = [
        (0.0 , {'P': [0.7943, 0.7597, 0.8081], 'R': [0.7889, 0.9368, 0.8825], 'F1': [0.7916, 0.8390, 0.8436]}),
        (0.1 , {'P': [0.5, 0.7, 0.81], 'R': [0.9, 0.3, 0.25], 'F1': [0.4, 0.03, 0.36]}),
        (0.2 , {'P': [0.4123, 0.592, 0.5132], 'R': [0.712, 0.212, 0.312], 'F1': [0.5123, 0.2, 0.46]})]
    x, y = list(zip(*test))
    x, y = list(x), list(y)
    Visualize(test_, metrics=["BERTScore", "ME", "BLEURT", "GRUEN"]).visualize()
