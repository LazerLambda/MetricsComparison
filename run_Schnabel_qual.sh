screen -dmS METhBERT_QUL_R
screen -S METhBERT_QUL_R -p 0 -X stuff 'python3.8 METh_BERT_entry.py -n 100 -sd 42 -s 4 -t MEBERT_QUL_100\n'

screen -dmS METhSBERT_QUL_R
screen -S METhSBERT_QUL_R -p 0 -X stuff 'python3.8 METh_SBERT_entry.py -n 2000 -sd 42 -s 4 -t MESBERT_QUL_2000\n'

screen -dmS MEOrgBERT_QUL_R
screen -S MEOrgBERT_QUL_R -p 0 -X stuff 'python3.8 MEOrg_BERT_entry.py -n 500 -sd 42 -s 4 -t MEOrgBERT_QUL_500\n'

screen -dmS MEOrgSBERT_QUL_R
screen -S MEOrgSBERT_QUL_R -p 0 -X stuff 'python3.8 MEOrg_SBERT_entry.py -n 2000 -sd 42 -s 4 -t MEOrgSBERT_QUL_2000\n'