from .Task import Task

from functools import reduce

import matplotlib.pyplot as plt

import copy
import math
import numpy as np
import pandas as pd
import random
import seaborn as sns
import spacy



class TwoDim(Task):

    # overwrite
    def set_steps(self, steps : dict) -> Task:
        step_txt : float = 1 / steps['txt']
        step_snt : float = 1 / steps['snt']
        self.step_arr.append(np.flip(1 - np.concatenate( (np.arange(0,1, step=step_txt), np.array([1])))))
        self.step_arr.append(np.flip(1 - np.concatenate( (np.arange(0,1, step=step_snt), np.array([1])))))
        return self

    @staticmethod
    def drop_single(sentence : str, doc : spacy.tokens.doc.Doc, step_snt : float) -> tuple:
        # TODO add upper bound for dropping

        bound : float = 1 - 1 / len(doc)
        if len(doc) == 0:
            return sentence, False

        candidates : list = []
        for i in range(len(doc)):
            if doc[i].pos_ != "PUNCT":
                candidates.append(i)
            else:
                continue

        # one word must be in the sentence at least
        if step_snt > bound:
            step_snt = bound

        prop : float = int(math.floor(step_snt * len(candidates)))
        drop_list = list = random.sample(candidates, k=prop)

        sent : str = ""

        for i in range(len(doc)):

            if i in drop_list:
                continue

            # TODO annotate
            token = doc[i]

            word = ""
            if i == 0:
                word = token.text
            else:
                if token.pos_ == "PUNCT":
                    word = token.text
                else:
                    word = " " + token.text
            sent += word
        return sent, True
    


    def perturbate(self) -> None:

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
                        new_sentence, success = self.drop_single(sentence=new_sentence, doc=doc[i], step_snt=step_snt)
                        if success:
                            indices.append(i)
                            sentences[i] = new_sentence

                    ret_tuple_snt[0].append(sentences)
                    ret_tuple_snt[1].append(indices)
                ret_txt.append(ret_tuple_snt)
            self.dmgd_texts.append(ret_txt)

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:

        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:

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
        
        self.df_sct = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree_txt', 'degree_snt', 'value'])

    def get_results(self) -> None:
        return self.df_sct.groupby(['metric', 'submetric', 'degree_txt', 'degree_snt']).mean()

    def plot(self, ax, title : str, metrics : list) -> None:
        submetric_list : list = reduce(lambda acc, elem: acc + elem, [metric.submetrics for metric in metrics], [])
        results = [(self.df_sct[self.df_sct['submetric'] == submetric].groupby(['metric', 'submetric', 'degree_txt', 'degree_snt'], as_index=False).mean()).pivot(index="degree_txt", columns="degree_snt", values="value") for submetric in submetric_list]
        for result in results:
            print(result)
            sns.heatmap(result, annot=True, fmt="g", cmap='viridis', ax=ax)
        ax.title.set_text(title)