from .Task import Task

from functools import reduce

import matplotlib.pyplot as plt

import copy
import math
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sns

from progress.bar import ShadyBar

class TwoDim(Task):

    #TODO slots
    #TODO subclass for drop
    #TODO description
    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]
    
    def __init__(self, params : dict):
        super(TwoDim, self).__init__(params)
        self.set_steps(params['steps'])

    # overwrite
    def set_steps(self, steps : dict) -> Task:
        step_txt : float = 1 / steps['txt']
        step_snt : float = 1 / steps['snt']
        self.step_arr.append(np.flip(1 - np.concatenate( (np.arange(0,1, step=step_txt), np.array([1])))))
        self.step_arr.append(np.flip(1 - np.concatenate( (np.arange(0,1, step=step_snt), np.array([1])))))
        return self

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:

        if len(metrics) == 0:
            return

        id_value : any = None
        bar : ShadyBar = ShadyBar(message="Evaluating " + self.name, max=len(self.step_arr[0]) * len(self.step_arr[1]) * len(self.texts))

        for i, _ in enumerate(self.step_arr[0]):
            step_results_txt : list = []
            for j, _ in enumerate(self.step_arr[1]):
                step_results_snt : list = []
                for k, (sentences, _) in enumerate(self.texts):
                    reference : list = sentences
                    candidate : list = None
                    if len(self.dmgd_texts[i][j][1][k]) == 0:
                        # Check if value for cand = ref already exists
                        if id_value == None:
                            # if it doesn't exist, assign cand = ref
                            candidate = sentences
                        else:
                            # if it exists, assign id value and continue
                            step_results_snt.append(id_value)
                            bar.next()
                            continue
                    else:     
                        candidate = self.dmgd_texts[i][j][0][k]

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

                    step_results_snt.append([*(res for res in self.__eval(reference, candidate, metrics))])

                    # if value for cand = ref doesn't exist, assign it to id value
                    if id_value == None:
                        id_value = step_results_snt[len(step_results_snt) - 1]
                    
                    bar.next()

                step_results_txt.append(step_results_snt)
            self.results.append(step_results_txt)
        bar.finish()

    def combine_results(self, metrics : list) -> None:
        for outer_run in self.results:
            outer_acc : list = []
            for run in outer_run:
                acc = dict(zip([metric.name for metric in metrics], [dict(zip(metric.submetrics, [[] for _ in metric.submetrics])) for metric in metrics]))
                for result in run:
                    for i, metric in enumerate(metrics):
                        for j, submetric in enumerate(metric.submetrics):
                            acc[metric.name][submetric] += result[i][j]
                outer_acc.append(acc)
            self.combined_results.append(outer_acc)

    def create_table(self, metrics : list) -> None:

        data : list = []
        for i, step_txt in enumerate(self.step_arr[0]):
            for j, step_snt in enumerate(self.step_arr[1]):
                for metric in metrics:
                    for submetric in metric.submetrics:
                        for value in self.combined_results[i][j][metric.name][submetric]:
                            scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree_txt' : float(step_txt), 'degree_snt' : float(step_snt), 'value' : float(value)}
                            data.append(scatter_struc)
        
        self.df = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree_txt', 'degree_snt', 'value'])

        # TODO
        # f = open(self.path + self.name + "_results_table_data.p", 'wb')
        # pickle.dump(self.df, f)
        # f.close()

    def get_results(self) -> None:
        return self.df.groupby(['metric', 'submetric', 'degree_txt', 'degree_snt']).mean()

    def plot(self, ax, title : str, metrics : list) -> None:
        submetric_list : list = reduce(lambda acc, elem: acc + list(zip(elem.submetrics, [elem.name for _ in elem.submetrics])), [metric for metric in metrics], [])
        metrics_dict : dict = dict(zip([metric.name for metric in metrics], metrics))
        results = [(self.df[self.df['submetric'] == submetric].groupby(['metric', 'submetric', 'degree_txt', 'degree_snt'], as_index=False).mean())
            .pivot(index="degree_txt", columns="degree_snt", values="value")\
                for submetric, _ in submetric_list]
        results: tuple = results, submetric_list
        for i, result in enumerate(results[0]):
            metric = results[1][i][1]
            vis_data = metrics_dict[metric].get_vis_info(self)
            sns.heatmap(
                result,
                annot=True,
                fmt="g",
                cmap=vis_data['color'],
                vmin=vis_data['vmin'],
                vmax=vis_data['vmax'],
                ax=ax[i])
            ax.flat[i].title.set_text(results[1][i][0])
        plt.suptitle(self.descr)