from __future__ import annotations


import os, datetime
import shutil


from Experiment import Experiment

class Comparison:

    __slots__ = ["experiment", "exp_wd"]

    def __init__(self) -> Comparison:
        self.experiment : Experiment = Experiment()
        pass

    def set_dir(self, name : str, loc : str = None) -> Comparison:

        if loc == ".":
            pass
        else:
            # TODO annotate

            self.exp_wd = os.path.join(os.getcwd(), name + "_" +datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(self.exp_wd)

        return self



    def config_exp(self,
            n : int,\
            degrees :int,\
            data_set_name : str = 'cnn_dailymail',\
            data_set_vers : str = '3.0.0',\
            perturbations : list = [
                'negated',
                'word_drop',
                'word_drop_every_sentence',
                'word_swap',
                'word_swap_every_sentence',
                'pos_drop_adj',
                'pos_drop_det',
                'repetitions']):

        self.experiment.sample(n)
        self.experiment.set_degrees(degrees)

    def create_data(self):
        self.experiment.perturb_data()

    def eval(self):
        pass

    def pipeline(self):
        pass

    def create_plot(self):
        pass

    def return_results(self, plot : bool = True, save : bool = True):
        pass

    def remove_folder(self):
        shutil.rmtree(self.exp_wd)



if  __name__ == "__main__":
    Comparison().set_dir("test")#.remove_folder()