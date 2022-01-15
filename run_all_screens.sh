
screen -dmS METhBERT_R
screen -S METhBERT_R -p 0 -X stuff 'python3.8 main.py -m MEMetricThBERT -st -n 2000 -sd 42 -s 4 -t MEBERT50 -sc 50\n'

screen -dmS BLEURT_R
screen -S BLEURT_R -p 0 -X stuff 'python3.8 main.py -m BLEURTRec -n 2000 -sd 42 -s 4 -t BLEURT2000\n'

screen -dmS BLEU_R
screen -S BLEU_R -p 0 -X stuff 'python3.8 main.py -m BLEUScoreMetric -n 2000 -sd 42 -s 4 -t BLEU2000\n'

screen -dmS BERTScore_R
screen -S BERTScore_R -p 0 -X stuff 'python3.8 main.py -m BERTScoreMetric -n 2000 -sd 42 -s 4 -t BERTScore2000\n'

screen -dmS BERTScoreIDF_R
screen -S BERTScoreIDF_R -p 0 -X stuff 'python3.8 main.py -m BERTScoreIDFMetric -n 2000 -sd 42 -s 4 -t BERTScoreIDF2000\n'

screen -dmS NUBIA50
screen -S NUBIA50 -p 0 -X stuff 'python3.8 main.py -m NUBIAMetric -n 2000 -sd 42 -s 4 -t NUBIA50 -sc 50\n'

screen -dmS MoverScore
screen -S MoverScore -p 0 -X stuff 'python3.8 main.py -m MoverScoreMetric -n 2000 -sd 42 -s 4 -t MoverScoreMetric2000\n'
