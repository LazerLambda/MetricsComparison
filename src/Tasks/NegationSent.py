from .OneDim import OneDim
from .Task import Task

import copy
import math
import numpy as np
import spacy

from checklist.perturb import Perturb


class Negation_Sent(OneDim, Task):

    @staticmethod
    def negate(sentence : str, doc : spacy.tokens.doc.Doc) -> tuple:
        success : bool = False
        try:
            ret = Perturb.perturb([doc], Perturb.add_negation, keep_original=False)
            if len(ret.data) > 0:
                sentence = ret.data[0][0]
                success = True
        except Exception:
            # if verbose:
            #     print("Failed to negate sentence {}".format(i))
            pass

        return sentence, success

    def perturbate(self) -> None:
        self.perturbate_1d(self.negate)

    