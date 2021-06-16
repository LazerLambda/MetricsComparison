from __future__ import annotations

import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import os
import spacy

from datasets import load_dataset, dataset_dict
from pathlib import Path
from src.Plot import Plot

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from typing import IO

class Experiment:
    # TODO slots

    def __init__(self, name : str = None, loc : str = None, verbose : bool = True):

        self.tasks : list = []
        self.metrics : list = []
        self.data : list = []
        self.verbose : bool = verbose
        self.result_files : list = []
        
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
                print("Existing folder loaded.")
            else:
                raise Exception("ERROR:\n\t'-> Specified location does not exist.")

    def __load_dataset(self, name : str, vers : str, n : int, seed : int = None) -> None:

        np.random.seed(seed=seed)

        data : dataset_dict.DatasetDict = load_dataset(name, vers)
        sample : np.ndarray = np.random.choice(np.arange(len(data['train'])), n)
        self.data : list = data['train'][sample]['article'] # TODO make property generics
        print("Sample chosen: " + str(sample)) if self.verbose else None

        # TODO set constant
        self.__dump((sample, self.data, name, vers), os.path.join(self.exp_wd, "original_data.p"))

    def __check_file_exists(self, file : str) -> bool: 
        return file in [f for f in os.listdir(self.exp_wd) if os.path.isfile(os.path.join(self.exp_wd, f))]


    def setup(
            self,
            tasks : list,
            metrics : list,
            data_specs : dict = {'name': 'cnn_dailymail', 'version' : '3.0.0', 'n' : 2, 'seed' : None},
            steps : dict = {'steps': 1, 'txt': 1, 'snt' : 2},
            **kwargs) -> Experiment:

        # TODO check if results are already available
        # [(task, (param_name, value))]

        nlp = spacy.load('en_core_web_sm')

        
        self.metrics = metrics

        # check if data is already existing
        if len(self.data) == 0:
            # TODO constant
            if self.__check_file_exists("original_data.p"):
                f : IO = open(os.path.join(self.exp_wd, "original_data.p"), 'rb')
                data : tuple = pickle.load(f)
                self.data : list = data[1]
                f.close()
                print("Existing data loaded.") if self.verbose else None
            else:
                self.__load_dataset(data_specs['name'], data_specs['version'], data_specs['n'], data_specs['seed'])

        assert len(self.data) != 0

        pos_list : list = []
        if 'pos_list' in kwargs:
            pos_list = kwargs['pos_list']

        tasks = [(e[0], []) if len(e) == 1 else e for e in tasks]
        tasks = [(task[0], {**{'data': self.data, 'nlp' : nlp, 'path' : self.exp_wd, 'steps' : steps, 'pos list': pos_list}, **dict(task[1])}) for task in tasks]
        
        # TODO sentence tokenization here
        # TODO spacy parsing here

        for task, param in tasks:
            self.tasks.append(task(param))

        return self

    def perturbate(self, overwrite : bool = False) -> Experiment:

        # TODO check if results are already existing

        for task in self.tasks:
            f_name : str = "." + task.name + "_perturbated_data.p"
            
            if self.__check_file_exists(f_name):
                print("Perturbations for " + task.name + " already exists.") if self.verbose else None
                f : IO = open(os.path.join(self.exp_wd, f_name), 'rb')
                data : list = pickle.load(f)
                f.close
                task.dmgd_texts = data
                continue

            task.perturbate()
            path : str = os.path.join(self.exp_wd, f_name)
            self.__dump(task.dmgd_texts, path)

    def __dump(self, data : any, path :str) -> None:

        f : IO = open(path, 'wb')
        pickle.dump(data, f)
        f.close()

    def evaluate(self, overwrite : bool = False) -> Experiment:

        pool = mp.Pool(len(self.tasks))
        metrics : list = self.metrics

        for task in self.tasks:

            f_name : str = "." + task.name + "_results_data.p"


            df_tmp : pd.DataFrame = pd.DataFrame()

            if self.__check_file_exists(f_name):
                f : IO = open(os.path.join(self.exp_wd, f_name), 'rb')
                df_tmp = pickle.load(f)
                # TODO consts
                metrics_in_df : np.ndarray = df_tmp['metric'].unique()
                print("Already computed: ", *(m for m in metrics_in_df)) if self.verbose else None
                metrics = [m for m in metrics if m.name not in metrics_in_df]
                print("New metrics to compute: ", *(m.name for m in metrics)) if self.verbose else None
                f.close()


            pool.apply(task.evaluate, args=[metrics])
            # task.evaluate(metrics)

        pool.close()
        pool.join()

        # added
        for task in self.tasks:

            f_name : str = "." + task.name + "_results_data.p"

            task.combine_results(self.metrics)
            task.create_table(self.metrics)

            
            df : pd.DataFrame = task.df
            if df_tmp.size != 0:
                df_tmp = df_tmp.append(df)
                df = df_tmp

                # assign all computed values to task
                task.df = df

            path : str = os.path.join(self.exp_wd, f_name)
            self.__dump(df, path)
            self.result_files.append(path)

            assert task.df.size != 0

    def plot(self, p_list : list, metrics : list) -> Experiment:

        assert isinstance(p_list, list) and isinstance(metrics, list)
        
        for p_class in p_list:
            p : Plot = p_class([(t, t.df, t.name, t.descr) for t in self.tasks], self.exp_wd)
            p.plot(metrics)
        pass

        
