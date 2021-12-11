
screen -dmS METhBERT_R
screen -S METhBERT_R -p 0 -X stuff 'python3 main.py -m MEMetricThBERT -st -n 500 -sd 42 -s 4 -t MEBERT500\n'

screen -dmS METhSBERT_R
screen -S METhSBERT_R -p 0 -X stuff 'python3 main.py -m MEMetricThSBERT -n 2000 -sd 42 -s 4 -t MESBERT2000\n'

screen -dmS MEOrgBERT_R
screen -S MEOrgBERT_R -p 0 -X stuff 'python3 main.py -m MEMetricOrgBERT -st -n 100 -sd 42 -s 4 -t MEOrgBERT100\n'

screen -dmS MEOrgSBERT_R
screen -S MEOrgSBERT_R -p 0 -X stuff 'python3 main.py -m MEMetricOrgSBERT -n 2000 -sd 42 -s 4 -t MEOrgSBERT2000\n'

screen -dmS BLEURT_R
screen -S BLEURT_R -p 0 -X stuff 'python3 main.py -m BLEURTRec -n 2000 -sd 42 -s 4 -t BLEURT2000\n'

screen -dmS BLEU_R
screen -S BLEU_R -p 0 -X stuff 'python3 main.py -m BLEUScoreMetric -n 2000 -sd 42 -s 4 -t BLEU2000\n'

screen -dmS BERTScore_R
screen -S BERTScore_R -p 0 -X stuff 'python3 main.py -m BERTScoreMetric -n 2000 -sd 42 -s 4 -t BERTScore2000\n'

screen -dmS BERTScoreIDF_R
screen -S BERTScoreIDF_R -p 0 -X stuff 'python3 main.py -m BERTScoreIDFMetric -n 2000 -sd 42 -s 4 -t BERTScoreIDF2000\n'

screen -dmS NUBIA
screen -S NUBIA -p 0 -X stuff 'python3 main.py -m NUBIAMetric -n 2000 -sd 42 -s 4 -t NUBIA2000\n'