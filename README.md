# Comparison of evaluation measures for generated text based on pre-trained language models

Experimental framework for the thesis "Comparison of evaluation measures for generated text based on pre-trained language models".

## Experiment-Project

- Experiment.py:
    - Handling of artificial impairment, evaluation and ploting (not used here)
- ExperimentMEOrg.py:
    - Inherited from Experiment.py to choose sentences instead of texts.
- Tasks/:
    - Task.py: Base class for all other tasks
    - OneDim.py: Base class for all other tasks, where only one dimension of damages is applied
    - TwoDim.py: Base class for DropAndSwap.py to impair in different ways
    - *.py: One dimensional tasks inheriting from OneDim
- metrics/:
    - Metric.py: Base class for all metrics, including specific information regarding the behaviour of the metric
    - *.py: Inheriting form Metric.py.
- Plot.py
    - Plots the results by task
- PlotByMetric:
    - inherits from Plot.py, plots the results by metric

- *entry.py:
    - modified main.py entry points for the GNU screen version in run_all_screens.sh
- main.py
    - Entry point for the experiment

## Mark-Evaluate
 - The implemented version of Mark-Evaluate is available at src/ME/

## Prerequisites
- python 3.8
- python libraries:
    ``` pip install --upgrade pip
        pip install -r requirements.txt```
- Specific data for spacy and nltk is required:
    ```
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt')"
    ```
- unzip installed 
- GNU screen install ed (required for running the experiment from the thesis)

- An installation script is available as install.sh
    ``` chmod +x install.sh
        ./install.sh
    ```

- Fine-tuned BERT on MNLI (available at: 'https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip') 
    must be placed in the src/ME/markevaluate folder s.t. 
    src/ME/markevaluate/bert_base_mnli/config.json eval_results.txt pytorch_model.bin vocab.txt

- BERTScore (https://github.com/Tiiiger/bert_score) must be downloaded into src/ and installed as described in the BERTScore README

- BLEURT (https://github.com/google-research/bleurt) must be downloaded into src/ and installed as described in the BLEURT README

## Re-Running the experiment

The experiment was run using the "run_all_screens.sh" script. This resutls will be written in hidden folders bearing the specific name of the experiment run.
    


## Visualizing the results

- To visualize the results, all folders with examples must be collected and moved to another folder, called 'outputs' in the root folder of the project.
  Another folder 'figures' must be created in the root directory of the project the Jupyter-Notebook plot.ipynb can then create the plots and the tables.
  The obtained figure will be stored in the figures folder.