from markevaluate import MarkEvaluate as me
from bert_score import BERTScorer
from bleurt import score
from GRUEN import Main


class TestSuite:

    def __init__(self, cand : list, ref : list) -> None:
        self.cand = cand
        self.ref = ref

    # scorer_bleurt : score.BleurtScorer = score.BleurtScorer(checkpoint="bleurt/bleurt/test_checkpoint") # checkpoint
    # scores_bleurt : list = scorer_bleurt.score(references=ref, candidates=cand)
    # print("BLEURT: ", scores_bleurt)

    # scores_gruen = Main.get_gruen(cand)
    # print("GRUEN: ", scores_gruen)

    def comp_ME(self) -> dict:
        me_main : me = me.MarkEvaluate(self.cand, self.ref)
        result_me : dict = me_main.estimate()
        return result_me

    def comp_BERTScore(self) -> dict:
        scorer_bertscore : BERTScorer = BERTScorer(lang="en")
        scores_bertscore : list = scorer_bertscore.score(self.cand, self.ref) # actual type is tensor
        return {"P": scores_bertscore[0],\
                "R": scores_bertscore[1],\
                "F1": scores_bertscore[2]}

    def comp_BLEURT(self) -> dict:
        return None

    def comp_GRUEN(self) -> dict:
        return None

    def comp_NUBIA(self) -> dict:
        return None
