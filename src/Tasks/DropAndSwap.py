from .TwoDim import TwoDim

import copy
import math
import numpy as np
import pandas as pd
import random
import seaborn as sns
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb



class DropAndSwap(TwoDim):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr", "nlp", "verbose"]

    def __init__(self, params : dict):
        super(DropAndSwap, self).__init__(params=params)
        self.name : str = "drop_and_swap"
        self.descr : str = "Dropped and swapped words"
        self.nlp : str = spacy.load('en_core_web_sm')
        self.verbose : bool = False

    @staticmethod
    def swap_pairs(sentence : str, doc : spacy.tokens.doc.Doc, step_swp : float) -> tuple:

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

        step_swp = 0.999999 if step_swp == 1.0 else step_swp
        upper : int = math.floor(step_swp * len(candidates))
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

    @staticmethod
    def drop_single(sentence : str, doc : spacy.tokens.doc.Doc, step : float) -> tuple:
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

        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr[0]) * len(self.step_arr[1]) * len(self.texts))
    
        for step_drop in self.step_arr[0]:
            ret_txt : list = []
            for step_swp in self.step_arr[1]:
                ret_tuple_snt : tuple = ([], [])
                for _, (sentences, doc) in enumerate(self.texts):
                    # sample : int = int(math.floor(step_txt * len(sentences)))

                    sentences : list = copy.deepcopy(sentences)
                    indices : list = []

                    if step_swp == 0.0 or step_drop == 0.0:
                        ret_tuple_snt[0].append([])
                        ret_tuple_snt[1].append([])
                        bar.next()
                        continue

                    for i in range(len(sentences)):

                        if len(doc[i]) < 2:
                            continue

                        new_sentence = sentences[i]
                        new_sentence, success = self.drop_single(sentence=new_sentence, doc=doc[i], step=step_drop)
                        if success:
                            new_sentence_swapped, success = self.swap_pairs(sentence=new_sentence, doc=self.nlp(new_sentence), step_swp=step_swp)
                            if success:
                                sentences[i] = new_sentence_swapped
                            else:
                                print("Sentence:\n'%s'\ncan't be deteriorated further." % new_sentence) if self.verbose else None
                                sentences[i] = new_sentence
                            indices.append(i)

                    ret_tuple_snt[0].append(sentences)
                    ret_tuple_snt[1].append(indices)
                    bar.next()
                ret_txt.append(ret_tuple_snt)
            self.dmgd_texts.append(ret_txt)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:

        if len(metrics) == 0:
            return

        id_value : any = None
        bar : ShadyBar = ShadyBar(message="Evaluating " + self.name, max=len(self.step_arr[0]) * len(self.step_arr[1]) * len(self.texts))

        for i, _ in enumerate(self.step_arr[0]):
            step_results_txt : list = []
            for j, _ in enumerate(self.step_arr[1]):
                step_results_snt : list = []
                for k, (sentences, _) in enumerate(self.texts):

                    
                    reference : list = []
                    candidate : list = []

                    if i == 0 or j == 0 or len(self.dmgd_texts[i][j][1][k]) == 0:
                        reference = sentences
                        candidate = self.dmgd_texts[i][j][0][k]
                    else:
                        indices : np.ndarray = np.asarray(self.dmgd_texts[i][j][1][k])

                        reference : list = np.asarray(sentences)[indices]
                        candidate : list = np.asarray(self.dmgd_texts[i][j][0][k])[indices]

                    # lookup the indices of the deteriorated text
                    if len(self.dmgd_texts[i][j][1][k]) == 0:
                        # Check if value for cand = ref already exists
                        if id_value == None:
                            # if it doesn't exist, assign cand = ref
                            candidate = sentences
                        else:
                            # if it exists, assign id value and continue
                            step_results_snt.append(id_value)
                            bar.next()
                            continue
                    else:     
                        candidate = self.dmgd_texts[i][j][0][k]

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

                    step_results_snt.append([*(res for res in self.__eval(reference, candidate, metrics))])

                    # if value for cand = ref doesn't exist, assign it to id value
                    if id_value == None:
                        id_value = step_results_snt[len(step_results_snt) - 1]
                    
                    bar.next()

                step_results_txt.append(step_results_snt)
            self.results.append(step_results_txt)
        bar.finish()

    def create_table(self, metrics : list) -> None:
        data : list = []
        for i, step_drp in enumerate(self.step_arr[0]):
            for j, step_swp in enumerate(self.step_arr[1]):
                for metric in metrics:
                    for submetric in metric.submetrics:
                        for value in self.combined_results[i][j][metric.name][submetric]:
                            scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree_drp' : float(step_drp), 'degree_swp' : float(step_swp), 'value' : float(value)}
                            data.append(scatter_struc)
        
        self.df = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree_drp', 'degree_swp', 'value'])

    def plot(self, ax : any, metric : any, submetric : str, **kwargs) -> None:
        result = self.df[self.df['submetric'] == submetric].groupby(['metric', 'submetric', 'degree_drp', 'degree_swp'], as_index=False)\
            .mean()\
            .pivot(index="degree_drp", columns="degree_swp", values="value")
        vis_data : dict = metric.get_vis_info(self)
        sns.heatmap(
            result,
            annot=True,
            fmt="g",
            cmap=vis_data['color'],
            vmin=vis_data['vmin'],
            vmax=vis_data['vmax'],
            cbar_kws={"shrink": 0.25},
            ax=ax)
        # ax.legend(bbox_to_anchor=(1,0), loc="lower left")#,  bbox_transform=fig.transFigure)
        # ax.set_ylabel("Degree of deterioration at text level", fontsize=10)
        # ax.set_xlabel("Degree of deterioration at sentence level", fontsize=10)