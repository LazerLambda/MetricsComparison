from .Metric import Metric
from bert_score import BERTScorer



class BERTScoreMetric(Metric):

    def __init__(self, name : str, description : str, submetric : list):
            
        self.scorer_bertscore: BERTScorer = BERTScorer(lang="en")
        super(BERTScoreMetric, self).__init__(name, description, submetric)


    def compute(self, ref : list, cand : list):
        #TODO check for symmetrie
        return self.scorer_bertscore.score(cand, ref)