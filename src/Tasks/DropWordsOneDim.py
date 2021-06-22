from .OneDim2 import OneDim2

import copy
import math
import numpy as np
import pandas as pd
import random
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb

class DropWordsOneDim(OneDim2):


    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(DropWordsOneDim, self).__init__(params=params)
        self.name = "dropped_words"
        self.descr = "Dropped words"

    @staticmethod
    def drop_single(sentence : str, doc : spacy.tokens.doc.Doc, step : float) -> tuple:
        # TODO add upper bound for dropping

        bound : float = 1 - 1 / len(doc)
        if len(doc) == 0:
            return sentence, False

        if step == 0:
            return sentence, True

        candidates : list = []
        for i in range(len(doc)):
            if doc[i].pos_ != "PUNCT":
                candidates.append(i)
            else:
                continue

        # one word must be in the sentence at least
        if step > bound:
            step = bound

        prop : float = int(math.floor(step * len(candidates)))
        drop_list : list = random.sample(candidates, k=prop)

        sent : str = ""

        for i in range(len(doc)):

            # exclude words to be dropped
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

        if len(sent) == 0:
            print("Sentence empty! Word drop")
            return sent, False

        return sent, True
    
    def perturbate(self) -> None:

        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr) * len(self.texts))

        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                for i in range(len(sentences)):
                    
                    if len(doc[i]) < 2:
                        continue

                    new_sentence = sentences[i]
                    new_sentence, success = self.drop_single(sentence=sentences[i], doc=doc[i], step=step)

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
    #         # if m.id and candidate == reference:
    #         #     print("HIER2")
    #         #     yield m.get_id(cand=candidate, ref=reference)
    #         # else:
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

    #             # if self.step_arr[i] == 0:
    #             #     # avoid computation for identity if identity is defined
    #             #     assert candidate == reference
    #             #     step_results.append([m.get_id(candidate, reference) for m in metrics])
    #             # else:
    #             step_results.append([*(res for res in self.__eval(reference, candidate, metrics))])
    #             bar.next()
    #         self.results.append(step_results)
    #     bar.finish()