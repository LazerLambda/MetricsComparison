from .Metric import Metric
from ..Tasks.Task import Task
from ..Tasks.OneDim import OneDim
from ..Tasks.TwoDim import TwoDim 
from bert_score import BERTScorer

import seaborn as sns


class BERTScoreMetric(Metric):

    limits : tuple = (0,1)

    def __init__(self, name : str, description : str, submetric : list):
        super(BERTScoreMetric, self).__init__(name, description, submetric)
            
        self.scorer_bertscore: BERTScorer = BERTScorer(lang="en")

        palette = sns.color_palette(None, 3)

        self.color : dict = {
            'R' : palette[0],
            'P' : palette[1],
            'F1' : palette[2]
        }
        self.id : tuple = True, 1


    def compute(self, ref : list, cand : list):
        #TODO check for symmetrie
        return self.scorer_bertscore.score(cand, ref)


    def get_vis_info(self, t : Task) -> dict():
        if isinstance(t, OneDim):
            return dict()

        if isinstance(t, TwoDim):
            return {
                'color': 'plasma',
                'vmin' : 0,
                'vmax' : 1
            }
        
        return None