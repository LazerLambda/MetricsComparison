from __future__ import annotations

import numpy as np
import pickle

from DamageData import DamageData
from datasets import load_dataset
from progress.bar import ShadyBar

class Experiment:

    def __init__(self, data_set_name : str = 'cnn_dailymail', data_set_vers : str = '3.0.0'):
         self.data_set = load_dataset(data_set_name, data_set_vers)



    def sample(self, sample_size) -> None:
        sample : np.ndarray = np.random.choice(np.arange(len(self.data_set['train'])), sample_size)
        self.data_ref = self.data_set['train'][sample]['article'] # TODO make property generics


    
    def perturb_data(self, steps : int) -> None:
        step_size : float = 1 / steps
        self.step_arr = np.flip(1 - np.arange(0,1, step=step_size))

        self.deteriorated_data : dict = {'negated' : [], 'word_drop' : [], 'word_swap' : [], 'pos_drop_adj' : [], 'pos_drop_det' : []}

        dd : DamageData = DamageData()

        bar = ShadyBar('Creating negated samples.', max=len(self.step_arr))

        for degree in self.step_arr:
            for text in self.data_ref:
                self.deteriorated_data['negated'].append(
                    (
                        degree,
                        dd.negate_data(text, degree)
                    )
                )
            bar.next()
        
        bar.finish()

        bar = ShadyBar('Creating word drop samples.', max=len(self.step_arr))

        for degree in self.step_arr:
            for text in self.data_ref:
                self.deteriorated_data['word_drop'].append(
                    (
                        degree,
                        dd.word_drop(text, degree)
                    )
                )
            bar.next()
        
        bar.finish()

        bar = ShadyBar('Creating word swap samples.', max=len(self.step_arr))

        for degree in self.step_arr:
            for text in self.data_ref:
                self.deteriorated_data['word_swap'].append(
                    (
                        degree,
                        dd.word_swap(text, degree)
                    )
                )
            bar.next()
        
        bar.finish()

        bar = ShadyBar('Creating ADJ POS dropped samples.', max=len(self.step_arr))

        for degree in self.step_arr:
            for text in self.data_ref:
                self.deteriorated_data['pos_drop_adj'].append(
                    ({
                        degree : dd.pos_drop(text, degree, 'ADJ')
                    })
                )
            bar.next()
        
        bar.finish()

        bar = ShadyBar('Creating DET POS dropped samples.', max=len(self.step_arr))

        for degree in self.step_arr:
            for text in self.data_ref:
                self.deteriorated_data['pos_drop_det'].append(
                    (
                        degree,
                        dd.pos_drop(text, degree, 'DET')
                    )
                )
            bar.next()
        
        bar.finish()


        pickle.dump(self.deteriorated_data, open( "deteriorated_data.p", "wb" ) )








# if __name__ == "__main__":
#     exp : Experiment = Experiment()
#     exp.sample(2)
#     exp.perturb_data(10)
#     # print(exp.deteriorated_data)