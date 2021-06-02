from __future__ import annotations


import os, datetime
import shutil


from pathlib import Path
from src.Experiment import Experiment
from src.Metrics import Metrics as mtrc
from src.ReduceData import ReduceData
from src.Visualize import Visualize


class Comparison:

    __slots__ = ["experiment", "exp_wd", "rd"]

    def __init__(self) -> Comparison:
        self.experiment : Experiment = Experiment()
        pass

    def set_dir(self, name : str = None, loc : str = None) -> Comparison:

        if loc == None:
            if name == None:
                name = "Comparison_gen_text_metrics"
            # TODO annotate
            self.exp_wd = os.path.join(os.getcwd(), name + "_" +datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(self.exp_wd)
        else:
            location = Path(loc)
            if location.is_dir():
                self.exp_wd = os.path.join(os.getcwd(), loc)
            else:
                raise Exception("ERROR:\n\t'-> Specified location does not exist.")
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
                'repetitions']) -> Comparison:

        self.experiment.load_dataset(data_set_name, data_set_vers)
        self.experiment.sample(n)
        self.experiment.set_degrees(degrees)
        return self

    def create_data(self) -> Comparison:
        self.experiment.perturb_data(loc=self.exp_wd)
        return self

    def evaluate(self, metrics : list) -> Comparison:
        self.experiment.evaluate(metrics, loc=self.exp_wd)
        return self

    def pipeline(self) -> Comparison:
        task_list = [
                "negated",
                "pos_drop_adj",
                "pos_drop_det",
                "repetitions",
                "word_drop",
                "word_drop_every_sentence",
                "word_swap",
                "word_swap_every_sentence"]
        self.rd = ReduceData([mtrc.BLEURT, mtrc.BERTSCORE])
        self.rd.add_data(task_list, folder_path=self.exp_wd)
        self.rd.vis_ready()
        return self

    def create_plot(self) -> Comparison:
        vis = Visualize(folder_path=self.exp_wd)
        # testVis.plot(test.plot_data, testVis.show_metrics_on_tasks)
        vis.plot(self.rd.plot_data, vis.show_tasks_by_metric)
        return self

    def return_results(self, plot : bool = True, save : bool = True):
        pass

    def remove_folder(self):
        shutil.rmtree(self.exp_wd)



# if  __name__ == "__main__":
#     Comparison().set_dir("test")#.remove_folder()
