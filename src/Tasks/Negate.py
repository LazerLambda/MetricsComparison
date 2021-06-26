from .OneDim import OneDim
from .Task import Task

import copy
import math
import numpy as np
import pandas as pd
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb


class Negate(OneDim, Task):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(Negate, self).__init__(params=params)
        self.name = "negation"
        self.descr = "Negated sentences in the text."
        self.step_arr = ['original', 'negated']

    @staticmethod
    def negate(sentence : str, doc : spacy.tokens.doc.Doc, **kwargs) -> tuple:
        success : bool = False
        try:
            ret = Perturb.perturb([doc], Perturb.add_negation, keep_original=False)
            if len(ret.data) > 0:
                sentence = ret.data[0][0]
                success = True
        except Exception:
            pass

        if len(sentence) == 0:
            print("Sentence empty! Negation.")
            return sentence, False

        return sentence, success

    def perturbate(self) -> None:
        # [(degree of deterioration, deteriorated text, indices)]

        # super(Negate, self).perturbate()

        # TODO remove
        self.step_arr = ['original', 'negated']
        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr) * len(self.texts))

        for step in range(len(self.step_arr)):
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                for i in range(step * len(sentences)):
                    
                    if len(doc[i]) < 2:
                        continue

                    new_sentence = sentences[i]
                    new_sentence, success = self.negate(sentence=sentences[i], doc=doc[i], step=None)

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence
                    
                
                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)
                bar.next()

            self.dmgd_texts.append(ret_tuple)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()

    # def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
    #     for m in metrics:
    #         yield m.compute(cand=candidate, ref=reference)


    # def evaluate(self, metrics : list) -> None:
    #     if len(metrics) == 0:
    #         return

    #     bar : ShadyBar = ShadyBar(message="Evaluating " + self.name, max=len(self.step_arr) * len(self.texts))
    #     for i, _ in enumerate(self.step_arr):
    #         step_results : list = []
    #         for j, (sentences, _) in enumerate(self.texts):
    #             reference : list = sentences
    #             candidate : list = self.dmgd_texts[i][0][j]

    #             # TODO into method
    #             # drop emtpy sentences
    #             ref_checked : list = []
    #             cand_checked : list = []

    #             # TODO
    #             for ref, cand in zip(reference, candidate):
    #                 if len(cand) != 0:
    #                     ref_checked.append(ref)
    #                     cand_checked.append(cand)
    #                 else:
    #                     continue
                
    #             reference = ref_checked
    #             candidate = cand_checked

    #             if self.step_arr[i] == 0:
    #                 assert candidate == reference
    #                 # step_results.append([m.get_id() for m in metrics])
    #             else:
    #                 step_results.append([*(res for res in self.__eval(reference, candidate, metrics))])
    #             bar.next()
    #         print(step_results)
    #         self.results.append(step_results)
    #     bar.finish()

    # override
    def create_table(self, metrics : list) -> None:
        data : list = []
        for i, step in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in self.combined_results[i][metric.name][submetric]:
                        # 'degree' is a string here
                        scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree' : str(step), 'value' : float(value)}
                        data.append(scatter_struc)
        
        self.df = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree', 'value'])

    