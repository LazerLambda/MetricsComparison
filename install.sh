
cd src

echo "Installing BLEURT"
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ..

echo "Installing BERTScore"
pip install bert-score

echo "Downloading BERT pre-trained on MNLI for Mark-Evaluate"
cd ME/markevaluate
mkdir bert_base_mnli
wget https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip
unzip MNLI_BERT.zip -d bert_base_mnli
rm MNLI_BERT.zip
# leaving markevaluates
cd ..
# leaving ME
cd ..
# leaving src
cd ..

echo "DONE"
