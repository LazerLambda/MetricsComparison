from .Task import Task

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from checklist.perturb import Perturb

class OneDim(Task):


    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:

        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:
        for i, _ in enumerate(self.step_arr):
            step_results : list = []
            for j, (sentences, _) in enumerate(self.texts):
                reference : list = sentences
                candidate : list = self.dmgd_texts[i][0][j]
                step_results.append([*(res for res in self.__eval(reference, candidate, metrics))])
            self.results.append(step_results)

    def combine_results(self, metrics : list) -> None:
        for run in self.results:
            acc = dict(zip([metric.name for metric in metrics], [dict(zip(metric.submetrics, [[] for _ in metric.submetrics])) for metric in metrics]))
            for result in run:
                for i, metric in enumerate(metrics):
                    for j, submetric in enumerate(metric.submetrics):
                        acc[metric.name][submetric] += result[i][j]
            self.combined_results.append(acc)

    def create_table(self, metrics : list) -> None:

        data : list = []
        for i, step in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in self.combined_results[i][metric.name][submetric]:
                        scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree' : float(step), 'value' : float(value)}
                        data.append(scatter_struc)
        
        self.df_sct = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree', 'value'])

    def get_results(self) -> None:
        return self.df_sct.groupby(['metric', 'submetric', 'degree']).mean()

    # TODO annotate
    def plot(self, ax, title : str) -> None:
        sns.set_theme(style="ticks", palette="pastel")
        sns.boxplot(x="degree", y="value",
            hue="submetric", # palette=["m", "g"],
            data=self.df_sct, ax=ax)
        ax.title.set_text(title)