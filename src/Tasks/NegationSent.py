from .Task import Task

import copy
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import random

from progress.bar import ShadyBar
from checklist.perturb import Perturb


class Negation_Sent(Task):

    def perturbate(self, params : dict, verbose : bool=False) -> None:

        n : int = params['n']
        steps : int = params['steps']
        step_size : float = 1 / steps
        self.step_arr = np.flip(1 - np.concatenate( (np.arange(0,1, step=step_size), np.array([1])) ) )


        # [(degree of deterioration, deteriorated text, indices)]
        self.dmgd_texts : list = []

        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for j, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                sample : int = int(math.floor(step * len(sentences)))

                for i in range(sample):
                    try:
                        ret = Perturb.perturb([doc[i]], Perturb.add_negation, keep_original=False)
                        if len(ret.data) > 0:
                            sentences[i] = ret.data[0][0]
                            indices.append(i)
                            continue
                    except Exception:
                        print("Failed to negate sentence {}".format(i)) if verbose else None
                
                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)

            self.dmgd_texts.append(ret_tuple)

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:

        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:
        self.results : list = []
        for i, _ in enumerate(self.step_arr):
            step_results : list = []
            for j, (sentences, _) in enumerate(self.texts):
                reference : list = sentences
                candidate : list = self.dmgd_texts[i][0][j]
                step_results.append([*(res for res in self.__eval(reference, candidate, metrics))])
            self.results.append(step_results)

    def combine_results(self, metrics : list) -> None:
        self.combined_results : list = []
        for run in self.results:
            acc = dict(zip([metric.name for metric in metrics], [dict(zip(metric.submetrics, [[] for _ in metric.submetrics])) for metric in metrics]))
            for result in run:
                for i, metric in enumerate(metrics):
                    for j, submetric in enumerate(metric.submetrics):
                        acc[metric.name][submetric] += result[i][j]
            self.combined_results.append(acc)

    def plot(self, fig : plt.figure) -> None:
        pass