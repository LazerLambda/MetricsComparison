from .Task import Task

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from checklist.perturb import Perturb
from progress.bar import ShadyBar

class OneDim(Task):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(OneDim, self).__init__(params=params)

        self.set_steps(params['steps'])

    def set_steps(self, steps : dict) -> Task:
        step_size : float = 1 / steps['steps']
        self.step_arr = np.flip(1 - np.concatenate( (np.arange(0,1, step=step_size), np.array([1])) ) )
        return self

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:
        
        if len(metrics) == 0:
            return

        bar : ShadyBar = ShadyBar(message="Evaluating " + self.name, max=len(self.step_arr) * len(self.texts))
        for i, _ in enumerate(self.step_arr):
            step_results : list = []
            for j, (sentences, _) in enumerate(self.texts):
                reference : list = sentences
                candidate : list = self.dmgd_texts[i][0][j]

                # TODO into method
                # drop emtpy sentences
                ref_checked : list = []
                cand_checked : list = []

                for ref, cand in zip(reference, candidate):
                    if len(cand) != 0:
                        ref_checked.append(ref)
                        cand_checked.append(ref)
                    else:
                        continue
                
                reference = ref_checked
                candidate = cand_checked

                if self.step_arr[i] == 0:
                    assert candidate == reference
                step_results.append([*(res for res in self.__eval(reference, candidate, metrics))])
                bar.next()
            self.results.append(step_results)
        bar.finish()

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
        
        self.df = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree', 'value'])

    def get_results(self) -> None:
        ret = self.df.groupby(['metric', 'submetric', 'degree']).mean()
        return ret

    # TODO annotate
    # TODO maybe different plots due to different scaling
    def plot(self, ax, title : str) -> None:
        sns.set_theme(style="ticks", palette="pastel")
        sns.boxplot(x="degree", y="value",
            hue="submetric", # palette=["m", "g"],
            data=self.df, ax=ax)
        ax.set(ylim=(-1.5, 1.5))
        ax.title.set_text(title)