from .OneDim import OneDim

import copy
import math
import numpy as np

from checklist.perturb import Perturb


class Negation_Sent(OneDim):

    def perturbate(self, params : dict, verbose : bool=False) -> None:
        # [(degree of deterioration, deteriorated text, indices)]

        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
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

    