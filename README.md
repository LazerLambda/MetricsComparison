# Comparison of evaluation measures for generated text based on pre-trained language models

Experimental framework for the paper [Pre-trained language models evaluating themselves - A comparative study](https://aclanthology.org/2022.insights-1.25/).


## Mark-Evaluate
 - The implemented version of Mark-Evaluate is available at src/metric/ME/

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
- GNU screen (required for running the experiment from the thesis)

- NUBIA: https://github.com/wl-research/nubia
- BERTScore: https://github.com/Tiiiger/bert_score
- MoverScore: https://github.com/AIPHES/emnlp19-moverscore
- BLEURT: https://github.com/google-research/bleurt
- Mark-Evaluate: https://github.com/LazerLambda/ME

- Fine-tuned BERT on MNLI (available at: 'https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip') 
    must be placed in the src/ME/markevaluate folder such that 
    src/ME/markevaluate/bert_base_mnli/config.json eval_results.txt pytorch_model.bin vocab.txt


## Re-Run the experiment

The experiment was run using the "run_all_screens.sh" script. This resutls will be written in hidden folders bearing the specific name of the experiment run.

## Experiment-Project

- src/Experiment.py:
    - Handling of artificial impairment and evaluation
- Tasks/:
    - Task.py: Base class for all other tasks
    - OneDim.py: Base class for all other tasks, where only one dimension of damages is applied
    - *.py: Specific tasks
- metrics/:
    - Metric.py: Base class for all metrics, including specific information regarding the behaviour of the metric
    - Custom metric folders must be stored here
    - custom_metrics/
        - *.py: Inheriting form Metric.py.
- Plot.py
    - Plots the results by task
- PlotByMetric:
    - inherits from Plot.py, plots the results by metric

- *entry.py:
    - modified main.py entry points for the GNU screen version in run_all_screens.sh
- main.py
    - Entry point for the experiment

## Folder Structure
- The folder `metrics` entails all data to compute the metrics
    - To evaluate each metric, a wrapper must be written based on the class  `Metric.py`
    - All custom metric wrapper files must be stored in the `custom metrics` folder
    - All metrics related files (e.g. repositories, models etc.) should be stored in the `metrics` folder so that:
    ```bash
    .
    ├── ...
    ├── metrics
    │   ├── RespectiveMetric1Folder     # Folder including prerequisites for metric_1_wrapper.py
    │   ├── RespectiveMetric2Folder     # Folder including prerequisites for metric_2_wrapper.py
    │   ├── Metric.py                   # Parent class for metric wrappers
    │   └── custom_etrics               # Put metric wrapper python files here
    │       ├── metric_1_wrapper.py
    │       ├── metric_2_wrapper.py
    │       └── ...
    └── ...
    ``` 

## Visualizing the results

- To visualize the results, all folders with examples must be collected and moved to another folder, called 'outputs' in the root folder of the project.
  Another folder 'figures' must be created in the root directory of the project the Jupyter-Notebook plot.ipynb can then create the plots and the tables.
  The obtained figure will be stored in the figures folder.
