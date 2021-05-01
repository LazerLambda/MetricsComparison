from markevaluate import MarkEvaluate as me
from bert_score import BERTScorer
from bleurt import score
from GRUEN import Main


cand , ref = ["asdf", "the thing goes skrrrrya pam pam pam pam pam", "and a kut tun tun boom yo"], ["asd", "The thing goes crazy.", "And makes some noise."]

me_main : me = me.MarkEvaluate(cand, ref)
result_me : dict = me_main.estimate()
print("Mark-Evaluate: ", result_me)

scorer_bertscore : BERTScorer = BERTScorer(lang="en")
scores_bertscore : list = scorer_bertscore.score(cand, ref) # actual type is tensor
print("BERTScore: ", scores_bertscore)

scorer_bleurt : score.BleurtScorer = score.BleurtScorer(checkpoint="bleurt/bleurt/test_checkpoint") # checkpoint
scores_bleurt : list = scorer_bleurt.score(references=ref, candidates=cand)
print("BLEURT: ", scores_bleurt)

scores_gruen = Main.get_gruen(cand)
print("GRUEN: ", scores_gruen)










