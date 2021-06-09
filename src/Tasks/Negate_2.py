from .OneDim import OneDim
from .Task import Task

import copy
import math
import numpy as np
import pandas as pd
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb


class Negate2(OneDim, Task):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(Negate2, self).__init__(params=params)
        self.name = "negation"
        self.descr = "Negated sentences with increasing degree in the text."

    @staticmethod
    def negate(sentence : str, doc : spacy.tokens.doc.Doc) -> tuple:
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

        self.step_arr = ['original', 'negated']
        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr) * len(self.texts))


        for step in [0, 1]:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                sample : int = int(math.floor(step * len(sentences)))

                for i in range(sample):
                    
                    if len(doc[i]) < 2:
                        continue

                    new_sentence = sentences[i]
                    new_sentence, success = self.negate(sentence=sentences[i], doc=doc[i])

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence
                    
                
                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)
                bar.next()

            self.dmgd_texts.append(ret_tuple)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        for m in metrics:
            id_b, id_v = m.id
            if id_b:
                yield id_v
            else:
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

    def create_table(self, metrics : list) -> None:
        data : list = []
        for i, step in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in self.combined_results[i][metric.name][submetric]:
                        scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree' : str(step), 'value' : float(value)}
                        data.append(scatter_struc)
        
        self.df = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree', 'value'])

    