from ..Metric import Metric
from bert_score import BERTScorer

from itertools import chain


class BERTScoreIDFMetric(Metric):

    limits : tuple = (0,1.05)

    def __init__(self):
        super(BERTScoreIDFMetric, self).__init__()

        # Properties
        self.name: str = "BERTScore (idf)"
        self.description: str = "BERTScore with idf-weighting"
        self.submetrics: list = ["R", "P", "F1"]
        self.id : bool = True

        
        self.scorer_bertscore: BERTScorer = None

    def set_exp(self, exp):
        super(BERTScoreIDFMetric, self).set_exp(exp)
        self.init_idf(
            [sentences for sentences, _ in self.experiment.texts])

    def init_idf(self, ref: list):
        """Initialize idf-weights.
        
        Reference corpus is passed as list of lists of strings
        from which the idf-weigthts are computed.
        """
        assert len(ref) != 0,\
            "Reference list is empty."
        assert isinstance(ref, list),\
            f"Passed value: {ref} is not of type 'list'"
        assert isinstance(ref[0], list),\
            f"First item: {ref[0]} of ref is not of type list"

        ref: list = list(chain.from_iterable(ref))
        self.scorer_bertscore: BERTScorer = BERTScorer(
            lang="en",
            idf=True,
            idf_sents=ref,
            rescale_with_baseline=True,
            use_fast_tokenizer=True)


    def get_id(self, ref :list, cand : list):
        assert len(ref) == len(cand)
        return ([1] * len(ref), [1] * len(ref), [1] * len(ref))


    def compute(self, ref : list, cand : list):
        assert len(ref) == len(cand)
        return self.scorer_bertscore.score(cand, ref)
