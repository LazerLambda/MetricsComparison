import math
import matplotlib.pyplot as plt
import seaborn as sns

from functools import reduce


class Plot:

    __slots__ = ["task_list"]

    def __init__(self, task_list : list):

        assert isinstance(task_list, list)
        assert False not in [len(e) == 4 for e in task_list]
        # TODO print specific error

        self.task_list : list = task_list
        pass

    @staticmethod
    def find_square(n : int) -> tuple:

        assert isinstance(n, int)
        
        x, y = math.floor(math.sqrt(n)), math.ceil(math.sqrt(n))

        while True:
            if x * y < n:
                y += 1
            else:
                return x, y

    def plot(self, metrics : list) -> None:

        assert isinstance(metrics, list)

        metric_list : list = reduce(lambda acc, metric: acc + list(zip(metric.submetrics, [metric for _ in metric.submetrics])), [metric for metric in metrics], [])

        palette = sns.color_palette(None, len(metric_list))

        for i, (task, df, name, descr) in enumerate(self.task_list):

            x, y = self.find_square(len(metric_list))
            _, axes = plt.subplots(nrows=y, ncols=x, figsize=(20, 10))

            for j, (submetric, metric) in enumerate(metric_list):

                x_i = j % x
                y_i = j // y

                ax = axes[y_i][x_i]

                task.plot(ax, metric, submetric)
                ax.legend(bbox_to_anchor=(1,0), loc="lower left")
                ax.title.set_text(metric.name + ": " + submetric if metric.name != submetric else metric.name)
                # ax.set_aspect('equal')
            
            plt.suptitle(descr)
            # plt.tight_layout()
            plt.show()