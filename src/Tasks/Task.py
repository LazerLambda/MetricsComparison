from __future__ import annotations


import copy
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import spacy

from functools import reduce

class Task():

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "df_sct", "step_arr", "path", "name"]

    def __init__(self, data: list, nlp: spacy.lang, path : str = ""):

        self.texts: list = []
        self.results : list = []
        self.combined_results : list = []
        self.step_arr : list = []
        self.dmgd_texts : list = []
        self.path : str = path

        for text in data:
            sentences: list = nltk.sent_tokenize(text)
            doc: list = list(nlp.pipe(sentences))
            self.texts.append((sentences, doc))

    def set_steps(self, steps : int) -> Task:
        step_size : float = 1 / steps
        self.step_arr = np.flip(1 - np.concatenate( (np.arange(0,1, step=step_size), np.array([1])) ) )
        return self

    def perturbate_1d(self, f : callable) -> None:
        # [(degree of deterioration, deteriorated text, indices)]

        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                sample : int = int(math.floor(step * len(sentences)))

                for i in range(sample):
                    
                    new_sentence = sentences[i]
                    new_sentence, success = f(sentence=sentences[i], doc=doc[i])

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence
                    
                
                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)

            self.dmgd_texts.append(ret_tuple)

    def perturbate_2d(self, f : callable) -> None:
    
        for step_txt in self.step_arr[0]:
            ret_txt : list = []
            for step_snt in self.step_arr[1]:
                ret_tuple_snt : tuple = ([], [])
                for _, (sentences, doc) in enumerate(self.texts):
                    sample : int = int(math.floor(step_txt * len(sentences)))

                    sentences : list = copy.deepcopy(sentences)
                    indices : list = []

                    if step_txt == 0.0 or step_snt == 0.0:
                        ret_tuple_snt[0].append([])
                        ret_tuple_snt[1].append([])
                        continue

                    for i in range(sample):

                        new_sentence = sentences[i]
                        new_sentence, success = f(sentence=new_sentence, doc=doc[i], step_snt=step_snt)
                        if success:
                            indices.append(i)
                            sentences[i] = new_sentence

                    ret_tuple_snt[0].append(sentences)
                    ret_tuple_snt[1].append(indices)
                ret_txt.append(ret_tuple_snt)
            self.dmgd_texts.append(ret_txt)

    def perturbate(self, params: dict) -> None:
        pass

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics: list) -> None:
        pass

    def combine_results(self, metrics : list) -> None:
        pass

    def evaluate_1d(self, metrics : list) -> None:
        for i, _ in enumerate(self.step_arr):
            step_results : list = []
            for j, (sentences, _) in enumerate(self.texts):
                reference : list = sentences
                candidate : list = self.dmgd_texts[i][0][j]
                if self.step_arr[i] == 0:
                    assert candidate == reference
                step_results.append([*(res for res in self.__eval(reference, candidate, metrics))])
            self.results.append(step_results)

    def evaluate_2d(self, metrics : list) -> None:

        id_value : any = None

        for i, step_txt in enumerate(self.step_arr[0]):
            step_results_txt : list = []
            for j, step_snt in enumerate(self.step_arr[1]):
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
                            continue
                    else:     
                        candidate = self.dmgd_texts[i][j][0][k]

                    step_results_snt.append([*(res for res in self.__eval(reference, candidate, metrics))])

                    # if value for cand = ref doesn't exist, assign it to id value
                    if id_value == None:
                        id_value = step_results_snt[len(step_results_snt) - 1]

                step_results_txt.append(step_results_snt)
            self.results.append(step_results_txt)

    def combine_results_1d(self, metrics : list) -> None:
        for run in self.results:
            acc = dict(zip([metric.name for metric in metrics], [dict(zip(metric.submetrics, [[] for _ in metric.submetrics])) for metric in metrics]))
            for result in run:
                for i, metric in enumerate(metrics):
                    for j, submetric in enumerate(metric.submetrics):
                        acc[metric.name][submetric] += result[i][j]
            self.combined_results.append(acc)

    def combine_results_2d(self, metrics : list) -> None:
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

    def create_table_1d(self, metrics : list) -> None:
        data : list = []
        for i, step in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in self.combined_results[i][metric.name][submetric]:
                        scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree' : float(step), 'value' : float(value)}
                        data.append(scatter_struc)
        
        self.df_sct = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree', 'value'])

    def create_table_2d(self, metrics : list) -> None:

        data : list = []
        for i, step_txt in enumerate(self.step_arr[0]):
            for j, step_snt in enumerate(self.step_arr[1]):
                for metric in metrics:
                    for submetric in metric.submetrics:
                        for value in self.combined_results[i][j][metric.name][submetric]:
                            scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree_txt' : float(step_txt), 'degree_snt' : float(step_snt), 'value' : float(value)}
                            data.append(scatter_struc)
        
        self.df_sct = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree_txt', 'degree_snt', 'value'])

        # TODO
        # f = open(self.path + self.name + "_results_table_data.p", 'wb')
        # pickle.dump(self.df_sct, f)
        # f.close()

    def get_results_1d(self) -> None:
        print(self.df_sct)
        # ret = self.df_sct.groupby(['metric', 'submetric', 'degree']).mean()
        ret = self.df_sct.groupby(['metric']).mean()
        return ret

    def get_results_2d(self) -> None:
        return self.df_sct.groupby(['metric', 'submetric', 'degree_txt', 'degree_snt']).mean()

    # TODO annotate
    def plot_1d(self, ax, title : str) -> None:
        sns.set_theme(style="ticks", palette="pastel")
        sns.boxplot(x="degree", y="value",
            hue="submetric", # palette=["m", "g"],
            data=self.df_sct, ax=ax)
        ax.title.set_text(title)

    def plot_2d(self, ax, title : str, metrics : list) -> None:
        submetric_list : list = reduce(lambda acc, elem: acc + list(zip(elem.submetrics, [elem.name for _ in elem.submetrics])), [metric for metric in metrics], [])
        metrics_dict : dict = dict(zip([metric.name for metric in metrics], metrics))
        results = [(self.df_sct[self.df_sct['submetric'] == submetric].groupby(['metric', 'submetric', 'degree_txt', 'degree_snt'], as_index=False).mean())
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
                ax=ax.flat[i])
            ax.flat[i].title.set_text(results[1][i][0])
        plt.suptitle(self.descr)

    def plot(self, fig: plt.figure) -> None:
        pass
