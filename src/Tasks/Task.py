import matplotlib.pyplot as plt
import nltk
import spacy



class Task():

    def __init__(self, text : str, nlp : spacy.lang.en.English):
        self.sentences = nltk.sent_tokenize(text)
        self.doc = list(nlp.pipe(self.sentences))
        pass

    def perturbate(self, params : dict) -> None:
        pass

    def evaluate(self, metrics : list) -> None:
        pass

    def plot(self, fig : plt.figure) -> None:
        pass