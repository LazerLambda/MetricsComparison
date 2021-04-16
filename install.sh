

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
mkdir cola_model
pip install -r requirements.txt
cd ..