"""This module plots different metrics by tasks."""

import math
import matplotlib.pyplot as plt
import os
import seaborn as sns

from functools import reduce


class Plot:
    """Class to organize plotting."""

    __slots__ = ["task_list", "wd"]

    def __init__(self, task_list: list, wd: str):
        """Do initialization of the class."""
        # check types
        assert isinstance(task_list, list)
        assert False not in [len(e) == 4 for e in task_list]
        # TODO print specific error

        self.task_list: list = task_list
        self.wd: str = wd
        pass

    @staticmethod
    def find_square(n: int) -> tuple:
        """Create tuple representating a square for plotting grid."""
        assert isinstance(n, int)

        x, y = math.floor(math.sqrt(n)), math.ceil(math.sqrt(n))

        while True:
            if x * y < n:
                y += 1
            else:
                return x, y

    def plot(self, metrics: list) -> None:
        """Plot data."""
        assert isinstance(metrics, list)

        metric_list: list = reduce(
            lambda acc, metric:
                acc + list(zip(
                    metric.submetrics,
                    [metric for _ in metric.submetrics])),
            [metric for metric in metrics], [])

        # palette = sns.color_palette(None, len(metric_list))

        for i, (task, df, name, descr) in enumerate(self.task_list):

            x, y = self.find_square(len(metric_list))
            _, axes = plt.subplots(
                nrows=y,
                ncols=x,
                figsize=(20, 20),
                subplot_kw={'aspect': 1})

            for j, (submetric, metric) in enumerate(metric_list):

                x_i = j % x
                y_i = j // x
                
                ax = None
                if x == 1:
                    ax = axes[y_i]
                else:
                    ax = axes[y_i][x_i]

                task.plot(ax, metric, submetric)
                ax.title.set_text(
                    metric.name + ": " + submetric if
                    metric.name != submetric else metric.name)
                ax.tick_params(axis='both', which='major', labelsize=8)

            plt.suptitle(descr)
            plt.tight_layout()
            path: str = os.path.join(self.wd, name + "_plot.png")
            plt.savefig(path)
            plt.show()
