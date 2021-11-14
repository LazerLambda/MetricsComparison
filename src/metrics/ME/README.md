# Mark-Evaluate 

Implementation of the [Mark-Evaluate method by Modido and Meinel (2020)](https://arxiv.org/abs/2010.04606) method to assess generated language.

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

The original version can be used by setting the original parameter to true `orig=True`

```python
    # orig BERT
    me_bert: ME = ME(sent_transf=False, sntnc_lvl=True, orig=True)
    # orig SBERT
    me_bert: ME = ME(orig=True)
``` 

## Tests:
```make test``` 