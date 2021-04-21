

pip install --upgrade pip

echo "Installing BLEURT"
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ..

echo "Installing BERTScore"
pip install bert-score

echo "Installing GRUEN"
git clone https://github.com/WanzhengZhu/GRUEN
cd GRUEN
pip install -r requirements.txt
python -m spacy download en_core_web_md
cd ..
echo "\n\nDownload CoLA classifier from this location: \033[94mhttps://drive.google.com/file/d/1Hw5na_Iy4-kGEoX60bD8vXYeJDQrzyj6/view\033[0m\n\t'-> \033[93mput the folder 'cola_model' in the GRUEN folder.\033[0m\n\n"