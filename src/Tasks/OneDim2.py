from .Task import Task
# from ..metrics import Metric

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from checklist.perturb import Perturb
from functools import reduce
from progress.bar import ShadyBar

class OneDim2(Task):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(OneDim2, self).__init__(params=params)

        self.set_steps(params['steps'])

    def set_steps(self, steps : dict) -> Task:
        step_size : float = 1 / steps['steps']
        self.step_arr = np.flip(1 - np.concatenate( (np.arange(0,1, step=step_size), np.array([1])) ) )
        return self

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        for m in metrics:
            if m.id and candidate == reference:
                yield m.get_id(cand=candidate, ref=reference)
            else:
                yield m.compute(cand=candidate, ref=reference)
        # for m in metrics:
        #     yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:

        if len(metrics) == 0:
            return

        bar : ShadyBar = ShadyBar(message="Evaluating " + self.name, max=len(self.step_arr) * len(self.texts))
        for i, _ in enumerate(self.step_arr):
            step_results : list = []
            for j, (sentences, _) in enumerate(self.texts):
                
                if i == 0 or len(self.dmgd_texts[i][1][j]) == 0:
                    reference : list = sentences
                    candidate : list = self.dmgd_texts[i][0][j]
                else:
                    indices : np.ndarray = np.asarray(self.dmgd_texts[i][1][j])
                    reference : list = np.asarray(sentences)[indices]
                    candidate : list = np.asarray(self.dmgd_texts[i][0][j])[indices]

                # TODO into method
                # drop emtpy sentences
                ref_checked : list = []
                cand_checked : list = []

                # TODO
                for ref, cand in zip(reference, candidate):
                    if len(cand) != 0:
                        ref_checked.append(ref)
                        cand_checked.append(cand)
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
    # def plot(self, ax : np.ndarray, title : str, metrics : list) -> None:
    #     submetric_list : list = reduce(lambda acc, elem: acc + list(zip(elem.submetrics, [elem.name for _ in elem.submetrics])), [metric for metric in metrics], [])
    #     results : list = [(self.df[self.df['submetric'] == submetric]) for submetric in submetric_list]
    #     results: tuple = results, submetric_list
    #     sns.set_theme(style="ticks", palette="pastel")
    #     print(ax)
    #     for i, result in enumerate(results[0]):
    #         sns.boxplot(x="degree", y="value",
    #             hue="submetric", # palette=["m", "g"],
    #             data=result, ax=ax[i])
    #         ax.set(ylim=(-1.5, 1.5))
    #         ax.title.set_text(title)

    def plot(self, ax : any, metric : any, submetric : str, **kwargs) -> None:

        # palette = ["m", "g"]

        # if 'color' in kwargs.keys():
        #     palette = [kwargs['color']]

        palette : list = [metric.color[submetric]]

        result = self.df[self.df['submetric'] == submetric]

        sns.set_theme(style="ticks", palette="pastel")
        sns.boxplot(
            x="degree",
            y="value",
            hue="submetric",
            palette=palette,
            data=result,
            ax=ax)
        ax.set(ylim=metric.limits)
        # ax.legend(bbox_to_anchor=(1,0), loc="lower right")#,  bbox_transform=fig.transFigure)
        ax.set_ylabel("Results")
        ax.set_xlabel("Degree of deterioration.", fontsize=10)
        ax.legend(bbox_to_anchor=(0,0), loc="lower left")
        