from __future__ import annotations

import nltk
import numpy as np
import pickle
import sys

from DamageData import DamageData
from datasets import load_dataset
from progress.bar import ShadyBar
from Metrics import Metrics

class Experiment:

    # TODO __slots__

    def __init__(self, data_set_name : str = 'cnn_dailymail', data_set_vers : str = '3.0.0'):
         self.data_set = load_dataset(data_set_name, data_set_vers) # TODO type
         self.data_ref : list = []
         self.metrics : Metrics = Metrics()



    def sample(self, sample_size) -> None:
        sample : np.ndarray = np.random.choice(np.arange(len(self.data_set['train'])), sample_size)
        self.data_ref = self.data_set['train'][sample]['article'] # TODO make property generics



    def apply_perturbations(self, type : str, message : str, perturb_fun : callable) -> None:
        bar = ShadyBar(message, max=len(self.step_arr))
        for i, degree in enumerate(self.step_arr):
            tmp_tpl : tuple = (degree, [])
            for text in self.data_ref:
                tmp_tpl[1].append(
                        perturb_fun(text, degree)
                )
            self.deteriorated_data[type].append(tmp_tpl)
            bar.next()
        bar.finish()



    def set_degrees(self, steps : int) -> None:
        step_size : float = 1 / steps
        self.step_arr = np.flip(1 - np.arange(0,1, step=step_size))
    


    def perturb_data(self) -> None:

        self.deteriorated_data : dict = {
            'ref' : self.data_ref,
            'negated' : [],
            'word_drop' : [],
            'word_swap' : [],
            'pos_drop_adj' : [],
            'pos_drop_det' : [],
            'repetitions' : []
            }

        dd : DamageData = DamageData()

        self.apply_perturbations('negated', 'Creating negated samples.', dd.negate_data)

        self.apply_perturbations('word_drop', 'Creating word drop samples.', dd.word_drop)

        self.apply_perturbations('word_swap', 'Creating word swap samples.', dd.word_swap)

        self.apply_perturbations('repetitions', 'Creating repetitions in samples.', dd.repeat_words)

        bar = ShadyBar('Creating ADJ POS dropped samples.', max=len(self.step_arr))
        for i, degree in enumerate(self.step_arr):
            tmp_tpl : tuple = (degree, [])
            for text in self.data_ref:
                tmp_tpl[1].append(
                        dd.pos_drop(text, degree, 'ADJ')
                )
            self.deteriorated_data['pos_drop_adj'].append(tmp_tpl)
            bar.next()
        bar.finish()


        bar = ShadyBar('Creating DET POS dropped samples.', max=len(self.step_arr))
        for i, degree in enumerate(self.step_arr):
            tmp_tpl : tuple = (degree, [])
            for text in self.data_ref:
                tmp_tpl[1].append(
                        dd.pos_drop(text, degree, 'DET')
                )
            self.deteriorated_data['pos_drop_det'].append(tmp_tpl)
            bar.next()
        bar.finish()

        pickle.dump(self.deteriorated_data, open("deteriorated_data.p", "wb" ))


    
    def evaluate_gen(self):

        for pert_type in self.deteriorated_data.keys():
            if pert_type == "ref":
                continue
            perturbations : list = self.deteriorated_data[pert_type]
            for i, degree in enumerate(self.step_arr):
                # tuple of (degree, array)
                cand_list : list = perturbations[i][1]
                # results : list = []
                for j, text in enumerate(self.data_ref):
                    # tuple of (perturbed text, indices of perturbed sentences)
                    cand : list = cand_list[j][0]
                    perturb_indcs : list = cand_list[j][1]
                    # results.append(Metrics.comp_BERTScore(nltk.sent_tokenize(text), cand))
                    ref_sentences : list = nltk.sent_tokenize(text)
                    yield (
                        self.metrics.comp_BERTScore(ref_sentences, cand),
                        self.metrics.comp_BLEURT(ref_sentences, cand),
                        self.metrics.comp_ME(ref_sentences, cand),
                        perturb_indcs
                        )
                    del cand
            # bar.finish()



    def eval(self) -> None:

        infile = open("deteriorated_data.p", "rb" )
        self.deteriorated_data = pickle.load(infile)
        self.data_ref = self.deteriorated_data['ref']

        for pert_type in self.deteriorated_data.keys():
            if pert_type == "ref":
                continue
            results_type : tuple = (pert_type, [])

            bar = ShadyBar(pert_type, max=len(self.step_arr) * len(self.data_ref))
            for i, degree in enumerate(self.step_arr):
                for j, text in enumerate(self.data_ref):
                    try:
                        results_type[1].append((degree, next(self.evaluate_gen())))
                        bar.next()
                    except StopIteration:
                        run = False
            print(results_type)
            pickle.dump(results_type, open(str(pert_type) + "_data.p", "wb" ))
            del results_type
            bar.finish()





if __name__ == "__main__":
    exp : Experiment = Experiment()
    exp.sample(1)
    exp.set_degrees(4)
    exp.perturb_data()
    exp.eval()
    # print(exp.deteriorated_data['word_swap'])g the Afghan Constitution and who are willing to come back willing to come back. "', 'Yet another piece of evidence about the folly of the U.S. involvement in Afghanistan came in the resignation of Matthew Hoh, a Foreign Service 