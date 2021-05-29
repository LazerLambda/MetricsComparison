

DESCRIPTION_BERTSCORE : dict = {
    'LEGEND' : ["P", "R", "F1"],
    'TITLE' : "BERTScore"
}

DESCRIPTION_BERTSCORE_F : dict = {
    'LEGEND' : ["P", "R", "F1"],
    'TITLE' : "BERTScore (Perturbed sentences only)"
}

DESCRIPTION_ME : dict = {
    'LEGEND' : ["Petersen", "Schnabel", "CAPTURE"],
    'TITLE' : "Mark-Evaluate"
}

DESCRIPTION_BLEURT : dict = {
    'LEGEND' : ["BLEURT"],
    'TITLE' : "BLEURT"
}

DESCRIPTION_BLEURT_F : dict = {
    'LEGEND' : ["BLEURT"],
    'TITLE' : "BLEURT (Perturbed sentences only)"
}


CONSTS : dict = {
    'ME' : DESCRIPTION_ME,
    'BERTScore' : DESCRIPTION_BERTSCORE,
    'BERTScore_f': DESCRIPTION_BERTSCORE_F,
    'BLEURT' : DESCRIPTION_BLEURT,
    'BLEURT_f' : DESCRIPTION_BLEURT_F
}