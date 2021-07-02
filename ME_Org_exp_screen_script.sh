
screen -dmS MEOrgBERT
screen -S MEOrgBERT -p 0 -X stuff 'python3.8 MEOrg_BERT_entry.py -n 500 -sd 42 -s 4 -t MEBERT100\n'

screen -dmS MEOrgSBERT
screen -S MEOrgSBERT -p 0 -X stuff 'python3.8 MEOrg_SBERT_entry.py -n 100 -sd 42 -s 4 -t MESBERT2000\n'
