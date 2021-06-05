from .TwoDim import TwoDim

import copy
import math
import spacy


class RepetitionTwoDim(TwoDim):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df_sct", "descr"]

    def __init__(self, data: list, nlp: spacy.lang, path : str = ""):
        self.name = "rep_words_2d"
        self.descr = "Reapeated word phrases in the text and with different intensity. Penalization"
        super(RepetitionTwoDim, self).__init__(data, nlp, path)


    @staticmethod
    def createRepetitions(\
            sentence : list,\
            doc : spacy.tokens.doc.Doc,\
            step_snt : float,\
            phraseLength : int = 4) -> bool:

        # subtract 1 because of indexing
        # end : int = len(doc) - 1
        # eos_index = len(doc) - 1
        for i in reversed(range(0, len(doc))):

            # find phrase at the end of the sentence without punctuation in it incrementally
            token_slice = doc[(i - phraseLength):i]
            # print([(token.text, token.pos_) for token in token_slice])
            for j in reversed(range(phraseLength)):
                j += 1
                token_slice = doc[(i - j):i]
                if not True in [token.pos_ == 'PUNCT' for token in token_slice]:
                    phraseLength = j
                    # break
                    token_slice = doc[(i - phraseLength):i]

                    acc : list = []
                    for k in range(i - phraseLength):
                        acc.append(doc[k])

                    n_times : int = math.floor(step_snt * len(doc))

                    acc += [token for token in token_slice] * n_times + [token for token in doc[i:len(doc)]]
                    
                    sent : str = ""

                    for i in range(len(acc)):


                        # TODO annotate
                        token = acc[i]

                        word = ""
                        if i == 0 or token.pos_ == "PUNCT":
                            word = token.text
                        else:
                            word = " " + token.text
                        sent += word
                    print(sent)

                    return sent, True
        
        return None, False


    def perturbate(self) -> None:
        self.perturbate_2d(self.createRepetitions)