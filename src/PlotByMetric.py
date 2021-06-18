"""This module plots different tasks by metric."""

from .Plot import Plot

import matplotlib.pyplot as plt
import os
import seaborn as sns

from functools import reduce


class PlotByMetric(Plot):
    """Class to organize plotting. Inherited from Plot class."""

    def __init__(self, task_list: list, wd: str):
        """Do initialization of the class."""
        super(PlotByMetric, self).__init__(task_list=task_list, wd=wd)

    def plot(self, metrics: list) -> None:
        """Plot data."""
        assert isinstance(metrics, list)

        metric_list: list = reduce(
            lambda acc, metric: acc +
            list(zip(
                metric.submetrics,
                [metric for _ in metric.submetrics])),
            [metric for metric in metrics], [])

        palette = sns.color_palette(None, len(self.task_list))

        for _, (submetric, metric) in enumerate(metric_list):

            # TODO annotate
            x, y = Plot.find_square(len(self.task_list))
            _, axes = plt.subplots(nrows=y, ncols=x, figsize=(20, 10))

            if y == 1:
                if x == 1:
                    axes = [[axes]]
                else:
                    axes = [axes]

            for j, (task, _, name, descr) in enumerate(self.task_list):

                x_i: int = j % x
                y_i: int = j // x

                # TODO annotate
                ax = axes[y_i][x_i]

                task.plot(ax, metric, submetric, color=palette[j])
                ax.legend(bbox_to_anchor=(1, 0), loc="lower left")
                ax.set_title(descr, fontweight="bold", size=10)

            plt.suptitle(metric.name)
            plt.tight_layout()
            path: str = os.path.join(self.wd, name + "_plot_by_metric.png")
            plt.savefig(path)
            plt.show()
