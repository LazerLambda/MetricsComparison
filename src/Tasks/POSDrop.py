from .OneDim import OneDim


import copy
import math
import spacy

class POSDrop(OneDim):

    def __init__(self, data: list, nlp: spacy.lang, path : str = "", pos : str = "ADJ"):
        self.name : str  = "rep_words_2d"
        self.descr : str  = "Reapeated word phrases in the text and with different intensity. Penalization"
        self.pos : str = pos
        super(POSDrop, self).__init__(data, nlp, path)

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
        

        return sentence, True

    def perturbate(self) -> None:
        self.perturbate_1d(self.drop_single_pos)
