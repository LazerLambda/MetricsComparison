# Mark-Evaluate 

Implementation of the [Mark-Evaluate (Petersen) method by Modido and Meinel (2020)](https://arxiv.org/abs/2010.04606) method to assess generated language.

The entry point of this project is MarkEvaluate.py in markevaluate.

Metric can be used in BERT-version and SBERT-version:

```python
    from markevaluate.MarkEvaluate import Markevaluate as ME

    # SBERT-version
    me_sbert: ME = ME()
    # BERT-version
    me_bert: ME = ME(sent_transf=False, sntnc_lvl=True)

    cand = ["This is a test."]
    ref = ["Hello World!"]

    me_bert.estimate(cand=cand, ref=ref)
    me_sbert.estimate(cand=cand, ref=ref)
```

## Tests:
```make test``` 