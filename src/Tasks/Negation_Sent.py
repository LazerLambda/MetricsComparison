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
        for ref, cand in zip(reference, candidate):
            for f in metrics:
                yield f(cand=cand, ref=ref)


    def evaluate(self, metrics : list) -> None:
        self.results : list = []
        for i, step in enumerate(self.step_arr):
            for j, text in enumerate(self.texts):
                reference : list = text
                candidate : list = self.dmgd_texts[i][0]
                self.results.append(*(res for res in self.__eval(reference, candidate, metrics)))

    def plot(self, fig : plt.figure) -> None:
        pass