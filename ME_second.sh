screen -dmS METhBERT_NEW_R
screen -S METhBERT_NEW_R -p 0 -X stuff 'python3.8 METh_BERT_entry.py -n 1000 -sd 42 -s 4 -t METhBERT_NEW_100\n'

screen -dmS METhSBERT_NEW_R
screen -S METhSBERT_NEW_R -p 0 -X stuff 'python3.8 METh_SBERT_entry.py -n 2000 -sd 42 -s 4 -t METhSBERT_NEW_2000\n'

screen -dmS MEOrgBERT_NEW_R
screen -S MEOrgBERT_NEW_R -p 0 -X stuff 'python3.8 MEOrg_BERT_entry.py -n 500 -sd 42 -s 4 -t MEOrgBERT_NEWL_500\n'

screen -dmS MEOrgSBERT_NEW_R
screen -S MEOrgSBERT_NEW_R -p 0 -X stuff 'python3.8 MEOrg_SBERT_entry.py -n 2000 -sd 42 -s 4 -t MEOrgSBERT_NEW_2000\n'