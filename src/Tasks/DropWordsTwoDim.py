from .TwoDim import TwoDim

import copy
import math
import random
import spacy

class DropWordsTwoDim(TwoDim):
    
    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(DropWordsTwoDim, self).__init__(params)
        self.name = "drop_words"
        self.descr = "Dropped words in sentences from the text and on sentence level."

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
        return sent, True
    
    def perturbate(self) -> None:
        self.perturbate_2d(self.drop_single)