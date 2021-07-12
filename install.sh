
cd src

echo "Installing python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading library specific data"
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"

echo "Installing BLEURT"
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ..

echo "Installing BERTScore"
pip install bert-score

cd ..
echo "\n\nDownload CoLA classifier from this location: \033[94mhttps://drive.google.com/file/d/1Hw5na_Iy4-kGEoX60bD8vXYeJDQrzyj6/view\033[0m\n\t'-> \033[93mput the folder 'cola_model' in the GRUEN folder.\033[0m\n\n"

cd ME/markevaluate
mkdir bert_base_mnli
unzip MNLI_BERT.zip -d bert_base_mnli
rm MNLI_BERT.zip
# leaving markevaluates
cd ..
# leaving ME
cd ..
# leaving src
cd ..

echo "DONE"
