from ..Metric import Metric
from bert_score import BERTScorer


class BERTScoreMetric(Metric):

    limits : tuple = (0,1.05)

    def __init__(self):
        super(BERTScoreMetric, self).__init__()

        # Properties
        self.name: str = "BERTScore"
        self.description: str = "BERTScore without idf-weighting"
        self.submetrics: list = ["R", "P", "F1"]
        self.id : bool = True

        self.scorer_bertscore: BERTScorer = BERTScorer(lang="en")


    def get_id(self, ref :list, cand : list):
        assert len(ref) == len(cand)
        return ([1] * len(ref), [1] * len(ref), [1] * len(ref))


    def compute(self, ref : list, cand : list):
        assert len(ref) == len(cand)
        return self.scorer_bertscore.score(cand, ref)