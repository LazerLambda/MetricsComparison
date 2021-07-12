# Comparison of evaluation measures for generated text based on pre-trained language models

## Mark-Evaluate
 - The implemented version of Mark-Evaluate is available at src/ME/

## Prerequisites
- python 3.8
- python libraries:
    ``` pip install --upgrade pip
        pip install -r requirements.txt```
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

 
    


## Visualizing the results

- To visualize the results, all folders with examples must be collected and moved to another folder, called 'outputs' in the root folder of the project.
  Another folder 'figures' must be created in the root directory of the project the Jupyter-Notebook plot.ipynb can then create the plots and the tables.
  The obtained figure will be stored in the figures folder.