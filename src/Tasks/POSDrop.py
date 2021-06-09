from .OneDim import OneDim


import copy
import math
import spacy

class POSDrop(OneDim):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr", "pos"]

    def __init__(self, params : dict):
        super(POSDrop, self).__init__(params=params)
        self.name : str  = "pos_drop"
        self.descr : str  = "Reapeated word phrases in the text and with different intensity. Penalization"
        self.pos : str = params['pos']

    def drop_single_pos(self, sentence : str, doc : spacy.tokens.doc.Doc) -> tuple:

        candidates : list = []

        for i in range(len(doc)):

            if doc[i].pos_ == self.pos:
                candidates.append(i)
            else:
                continue
        
        if len(candidates) == 0:
            return sentence, False
        
        diff : int = 0
        for i in candidates:
            bounds = doc[i].idx - diff, doc[i].idx + len(doc[i].text) - diff
            sentence = sentence[0:bounds[0]] + sentence[(bounds[1] + 1)::]
            diff += len(doc[i].text) + 1
        
        if len(sentence) == 0:
            print("Sentence empty! POS drop")
            return sentence, False

        return sentence, True

    def perturbate(self) -> None:
        self.perturbate_1d(self.drop_single_pos)
