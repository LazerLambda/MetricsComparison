from __future__ import annotations

import numpy as np
import pickle

from DamageData import DamageData
from datasets import load_dataset
from progress.bar import ShadyBar
from Metrics import Metrics

class Experiment:

    def __init__(self, data_set_name : str = 'cnn_dailymail', data_set_vers : str = '3.0.0'):
         self.data_set = load_dataset(data_set_name, data_set_vers) # TODO type



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


    
    def evaluate(self) -> None:
        infile = open("deteriorated_data.p", "rb" )
        self.deteriorated_data = pickle.load(infile)  

        for pert_type in self.deteriorated_data.keys():
            bar = ShadyBar(pert_type, max=len(self.step_arr))
            perturbations : list = self.deteriorated_data[pert_type]
            for i, degree in enumerate(self.step_arr):
                cand_list : tuple = perturbations[i]
                for j, text in enumerate(self.data_ref):
                    cand : list = cand_list[j]
                    # print(text, cand)
                    Metrics.comp_ME(text, cand)
                    bar.next()
            bar.finish()








if __name__ == "__main__":
    exp : Experiment = Experiment()
    exp.sample(2)
    exp.set_degrees(10)
    # exp.perturb_data()
    exp.evaluate()
    # print(exp.deteriorated_data['word_swap'])g the Afghan Constitution and who are willing to come back willing to come back. "', 'Yet another piece of evidence about the folly of the U.S. involvement in Afghanistan came in the resignation of Matthew Hoh, a Foreign Service 