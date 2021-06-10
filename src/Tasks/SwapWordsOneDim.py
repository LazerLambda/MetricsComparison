from .OneDim2 import OneDim2

import copy
import math
import numpy as np
import pandas as pd
import random
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb

class SwapWordsOneDim(OneDim2):


    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(SwapWordsOneDim, self).__init__(params=params)
        self.name = "swapped_words"
        self.descr = "Swapped words"

    @staticmethod
    def swap_pairs(sentence : str, doc : spacy.tokens.doc.Doc, step : float) -> tuple:

        # identity category
        if step == 0:
            return sentence, True

        candidates : list = []
        candidates_text : list = []

        for i in range(len(doc)):

            lower_text = doc[i].text.lower()

            # TODO maybe exclude first token ?

            if doc[i].pos_ != "PUNCT" and not lower_text in candidates_text:
                candidates.append(i)
                candidates_text.append(lower_text)
            else:
                continue

        if len(candidates) < 3:
            return sentence, False

        step = 0.999999 if step == 1.0 else step
        upper : int = math.floor(step * len(candidates))
        if upper % 2 == 1:
            upper += 1

        assert int(upper) <= len(candidates)

        sample : list = random.sample(candidates, upper)
        sample = [(sample[2 * i], sample[2 * i + 1]) for i in range(upper // 2)]

        tmp_l : list = []
        for i in range(len(doc)):
            tmp_l.append(doc[i])

        for i, j in sample:
            # TODO annotate
            tmp_t : any = tmp_l[i]
            tmp_l[i] = tmp_l[j]
            tmp_l[j] = tmp_t

        sent = ""
        for i, token in enumerate(tmp_l):

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
            print("Sentence empty! Word swap")
            return sent, False

        return sent, True
    
    def perturbate(self) -> None:

        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr) * len(self.texts))

        print(self.step_arr)
        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                for i in range(len(sentences)):
                    
                    if len(doc[i]) < 2:
                        continue

                    new_sentence = sentences[i]
                    new_sentence, success = self.swap_pairs(sentence=sentences[i], doc=doc[i], step=step)

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence
                    
                
                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)
                bar.next()

            self.dmgd_texts.append(ret_tuple)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()