from __future__ import annotations

import datetime
import numpy as np
import os
import spacy

from datasets import load_dataset, dataset_dict
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

class Experiment:
    # TODO slots

    def __init__(self, name : str = None, loc : str = None, verbose : bool = True):

        self.tasks : list = []
        self.metrics : list = []
        self.data : list = []
        self.verbose : bool = verbose
        
        # create directory if not specified
        if loc == None:
            # create name if not specified
            if name == None:
                name = "Comparison_gen_text_metrics"

            self.exp_wd : str = os.path.join(os.getcwd(), "." + name + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(self.exp_wd)
        else:
            location = Path(loc)
            if location.is_dir():
                self.exp_wd = os.path.join(os.getcwd(), loc)
            else:
                raise Exception("ERROR:\n\t'-> Specified location does not exist.")

    def __load_dataset(self, name : str, vers : str, n : int):
        data : dataset_dict.DatasetDict = load_dataset(name, vers)
        sample : np.ndarray = np.random.choice(np.arange(len(data['train'])), n)
        self.data : list = data['train'][sample]['article'] # TODO make property generics
        print("Sample chosen: " + str(sample)) if self.verbose else None


    def setup(
            self,
            tasks : list,
            metrics : list,
            data_specs : dict = {'name': 'cnn_dailymail', 'version' : '3.0.0', 'n' : 2},
            steps : dict = {'steps': 1, 'txt': 1, 'snt' : 2},
            visualizations : any = None) -> Experiment:

        # TODO check if results are already available
        # [(task, (param_name, value))]

        nlp = spacy.load('en_core_web_sm')

        
        self.metrics = metrics

        if len(self.data) == 0:
            self.__load_dataset(data_specs['name'], data_specs['version'], data_specs['n'])

        tasks = [(e[0], []) if len(e) == 1 else e for e in tasks]
        tasks = [(task[0], {**{'data': self.data, 'nlp' : nlp, 'path' : self.exp_wd, 'steps' : steps}, **dict(task[1])}) for task in tasks]

        for task, param in tasks:
            self.tasks.append(task(param))

        return self

    def perturbate(self, overwrite : bool = False) -> Experiment:

        # TODO check if results are already existing

        for task in self.tasks:
            task.perturbate()

        
    