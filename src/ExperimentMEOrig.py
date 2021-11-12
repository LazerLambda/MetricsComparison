

import Experiment
import datetime
import nltk
import numpy as np
import pandas as pd
import pickle
import os
import spacy

# from progress.bar import ShadyBar

class ExperimentMEOrig(Experiment.Experiment):
    """Specific class for Mark-Evaluate.
    
    Due to the computational costs, only sentences are choosen instead of 
    texts from the dataset.
    """
    def setup(
            self,
            tasks: list,
            metrics: list,
            data_specs: dict = {
                'name': 'cnn_dailymail',
                'version': '3.0.0',
                'n': 2,
                'f': (lambda data, sample: data['train'][sample]['article']),
                'seed': None},
            steps: dict = {
                'steps': 1,
                'txt': 1,
                'snt': 2},
            **kwargs) -> Experiment:
        """Initialize the experiment.
        
        Use sentences instead of the whole text
        """

        self.metrics = metrics

        # check if data already exists
        if len(self.data) == 0:
            # TODO constant
            # if super(ExperimentMEOrig, self).check_file_exists("original_data.p"):
            if self.check_file_exists("original_data.p"):
                f: IO = open(
                    os.path.join(
                        self.exp_wd,
                        "original_data.p"),
                    'rb')
                data: tuple = pickle.load(f)
                self.data: list = data[1]
                f.close()
                print("Existing data loaded.") if self.verbose else None
            else:
                self.load_dataset(
                    data_specs['name'],
                    data_specs['version'],
                    data_specs['n'],
                    data_specs['f'],
                    data_specs['seed'])

        assert len(self.data) != 0

        # only if pos tags are provided
        pos_list: list = []
        if 'pos_list' in kwargs:
            pos_list = kwargs['pos_list']

        # Prepare reference data
        self.texts: list = []
        nlp = spacy.load('en_core_web_sm')

        # bar: ShadyBar = ShadyBar("Creating dataset", max=len(self.data))

        sent_candidates: list = []
        for text in self.data:
            sentences: list = nltk.sent_tokenize(text)
            for sentence in sentences:
                doc = nlp(sentence)
                if len(doc) > 6 and len(doc) < 50:
                    sent_candidates.append(([sentence], [doc]))
            # bar.next()
        # bar.finish()
        
        sample: np.ndarray = np.random.choice(
            np.arange(
                len(sent_candidates)),
                data_specs['n'])

        for i in sample:
            self.texts.append(sent_candidates[i])

        # prepare task configuration
        # TODO standardize pos_list
        tasks = [(e[0], []) if len(e) == 1 else e for e in tasks]
        tasks = [(task[0], {**{
            # TODO rm data
            'data': self.data,
            'texts': self.texts,
            'nlp': nlp,
            'path': self.exp_wd,
            'steps': steps,
            'seed': data_specs['seed'],
            'pos list': pos_list},
            **dict(task[1])}) for task in tasks]

        # TODO sentence tokenization here
        # TODO spacy parsing here

        for task, param in tasks:
            self.tasks.append(task(param))

        return self