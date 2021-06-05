from .TwoDim import TwoDim

import copy
import math
import random
import spacy


class SwapWordsTwoDim(TwoDim):


    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df_sct", "descr"]

    def __init__(self, data: list, nlp: spacy.lang, path : str = ""):
        self.name = "swap_words"
        self.descr = "Swapped words in sentences from the text and on sentence level."
        super(SwapWordsTwoDim, self).__init__(data, nlp, path)

    @staticmethod
    def swap_pairs(sentence : str, doc : spacy.tokens.doc.Doc, step_snt : float) -> tuple:

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

        step_snt = 0.999999 if step_snt == 1.0 else step_snt
        upper : int = math.floor(step_snt * len(candidates))
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

        return sent, True

    def perturbate(self) -> None:
        self.perturbate_2d(self.swap_pairs)
