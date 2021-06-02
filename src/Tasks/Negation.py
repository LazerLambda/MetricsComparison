from .Task import Task

import matplotlib.pyplot as plt
import nltk

from checklist.perturb import Perturb


class Negation(Task):

    def perturbate(self, params : dict) -> None:
        percentage : float = params['p']


        if percentage < 0 or percentage > 1:
            raise Exception("ERROR: percentage must be in [0,1].")

        sentences = nltk.sent_tokenize(text)
        doc = list(self.nlp.pipe(sentences))

        loopCond : bool = True
        maxLen : int = len(sentences)
        counter : int = 0

        while loopCond:

            if counter > maxLen:
                loopCond = False

            sample = random.sample(range(len(sentences)), math.floor(percentage * len(sentences)))
            indices : list = []

            for i in sample:

                ret = None

                try:
                    ret = Perturb.perturb([doc[i]], Perturb.add_negation, keep_original=False)
                    if len(ret.data) > 0:
                        sentences[i] = ret.data[0][0]
                        indices.append(i)
                    else:
                        loopCond = False

                except Exception:
                    loopCond = True

                loopCond = False
            
            counter += 1

        return sentences, indices
        pass

    def evaluate(self, metrics : list) -> None:
        pass

    def plot(self, fig : plt.figure) -> None:
        pass