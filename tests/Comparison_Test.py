import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import numpy as np
import unittest

from Comparison import Comparison
from os import listdir
from os.path import isfile, join
from pathlib import Path
from src.Metrics import Metrics as mtrc


class Comparison_Test(unittest.TestCase):

    def test_file_handling(self):
        test = Comparison()
        test.set_dir()

        test_file = Path(test.exp_wd)
        self.assertTrue(test_file.is_dir())

        test.remove_folder()
        self.assertFalse(test_file.is_dir())

    def test_generation(self):
        test = Comparison()
        test.set_dir()
        test.config_exp(n=2, degrees=1)
        test.create_data()
        # test.evaluate(metrics)
        files = [f for f in listdir(test.exp_wd) if isfile(join(test.exp_wd, f))]
        self.assertTrue(['deteriorated_data_raw.p'] == files, msg="Check if folder looks correct")
        test.remove_folder()
        del test

    def test_evaluation(self):
        test = Comparison()
        test.set_dir(name=None, loc="test")

        metrics  = [ test.experiment.metrics.comp_BERTScore, test.experiment.metrics.comp_BLEURT,]
        perturbations = [
                'negated',
                'word_drop',
                'word_drop_every_sentence',
                'word_swap',
                'word_swap_every_sentence',
                'pos_drop_adj',
                'pos_drop_det',
                'repetitions']
        test.config_exp(n=2, degrees=1)
        # test.create_data()
        test.evaluate(metrics)
        # files = [f for f in listdir(test.exp_wd) if isfile(join(test.exp_wd, f))]
        # self.assertTrue(['deteriorated_data_raw.p'] == files, msg="Check if folder looks correct")
        test.evaluate()
        test.remove_folder()
        del test

    def test_plot(self):
        test = Comparison()
        test.set_dir(name=None, loc="test")

        test.pipeline().create_plot()

if __name__ == '__main__':
    unittest.main()