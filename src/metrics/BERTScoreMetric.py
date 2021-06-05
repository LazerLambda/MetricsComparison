from .Metric import Metric
from ..Tasks.Task import Task
from ..Tasks.OneDim import OneDim
from ..Tasks.TwoDim import TwoDim 
from bert_score import BERTScorer



class BERTScoreMetric(Metric):

    def __init__(self, name : str, description : str, submetric : list):
            
        self.scorer_bertscore: BERTScorer = BERTScorer(lang="en")
        super(BERTScoreMetric, self).__init__(name, description, submetric)


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