
import matplotlib.pyplot as plt
import nltk
import spacy


class Task():

    def __init__(self, data: list, nlp: spacy.lang):
        self.texts: list = []

        for text in data:
            sentences: list = nltk.sent_tokenize(text)
            doc: list = list(nlp.pipe(sentences))
            self.texts.append((sentences, doc))

    def perturbate(self, params: dict) -> None:
        pass

    def evaluate(self, metrics: list) -> None:
        pass

    def plot(self, fig: plt.figure) -> None:
        pass
