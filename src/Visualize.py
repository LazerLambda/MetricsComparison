
import numpy as np
import matplotlib.pyplot as plt


from cycler import cycler
from matplotlib.collections import EventCollection
from matplotlib.pyplot import figure
from .Metrics import Metrics as mtrc
from .ReduceData import ReduceData


class Visualize:

        def __init__(
                        self,
                        file_path : str = "",
                        color_arr: list = ['c', 'm', 'y', 'k'],
                        linestyle_arr: list = ['-', '--', ':', '-.']):

                self.color_arr: list = color_arr
                self.linestyle_arr: list = linestyle_arr
                # self.fig: plt.figure = plt.figure()
                self.default_cycler: cycler = (cycler(color=self.color_arr) +
                                        cycler(linestyle=self.linestyle_arr))


                self.plots : dict = {
                        'BERTScore' : dict(),
                        'BERTScore_f' : dict(),
                        'BLEURT' : dict(),
                        'BLEURT_f': dict(),
                        'ME' : dict() 
                }



        @staticmethod
        def create_plot(data : dict, task : str, metric : str, fig : plt.figure, title : str, subplot_info : tuple = (2, 3)):

                description : dict = data['description']
                x_data : np.ndarray = data['x']
                y_data_mean : np.ndarray = data['y_mean']
                y_data_min : np.ndarray = data['y_min']
                y_data_max : np.ndarray = data['y_max']

                # set up graph
                submetrics: int = np.shape(y_data_mean)[1]
                ax: plt.axes._subplots.AxesSubplot = fig.add_subplot(subplot_info[0], subplot_info[1], len(fig.axes) + 1)
                ax.set_xlim((-0.1, 1))
                ax.set_ylim((-0.1, 1.1))
                
                for j in range(submetrics):
                        # display mean
                        ax.scatter(x_data, y_data_mean[:, j])
                        ax.plot(x_data, y_data_mean[:, j])

                        # TODO display min max
                        # for x in x_data:
                        #         print(y_data_min[:, j][i])
                        #         ax.axvline(x=x,ymin=y_data_min[:, j][i],ymax=y_data_max[:, j][i],c=style_properties[j]['color'],linewidth=1,zorder=0, alpha=0.5)

                        # ax.boxplot(y_data_scatter)
                        # ax.set_xticklabels(x_data)
                ax.set_title(title)
                ax.legend(tuple(description['LEGEND']))



        def plot(self, data : dict, fun : callable):
                fun(data, self.create_plot)
                plt.show()



        def show_metrics_on_tasks(self, data : dict, plot : callable):

                for task in data.keys():
                        fig: plt.figure = plt.figure()
                        for score in data[task].keys():
                                plot(data[task][score], task, score, fig, data[task][score]['description']['title'])
                        plt.show()



        def show_tasks_by_metric(self, data : dict, plot : callable):
                
                scores : list =list(data[list(data.keys())[0]].keys())
                for s in scores:
                        fig: plt.figure = plt.figure()
                        fig.set
                        for task in data.keys():
                                for score in data[task].keys():
                                        if score == s:
                                                plot(data[task][score], task, score, fig, task,subplot_info=(3,3))
                        fig.suptitle(s)
                        plt.show()


if __name__ == "__main__":
        # import pickle
        file_list = [
                "negated_data.p",
                "pos_drop_adj_data.p",
                "pos_drop_det_data.p",
                "repetitions_data.p",
                "word_drop_data.p",
                "word_drop_every_sentence_data.p",
                "word_swap_data.p",
                "word_swap_every_sentence_data.p"]

        # test =ReduceData(example_data)
        task_list = [
                "negated",
                "pos_drop_adj",
                "pos_drop_det",
                "repetitions",
                "word_drop",
                "word_drop_every_sentence",
                "word_swap",
                "word_swap_every_sentence"]

        test = ReduceData([mtrc.BLEURT, mtrc.BERTSCORE])
        test.add_data(task_list, folder_path="./output_9/")
        test.vis_ready()

        testVis = Visualize()
        # testVis.plot(test.plot_data, testVis.show_metrics_on_tasks)
        testVis.plot(test.plot_data, testVis.show_tasks_by_metric)