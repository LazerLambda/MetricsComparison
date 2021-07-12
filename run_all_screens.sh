
screen -dmS METhBERT_R
screen -S METhBERT_R -p 0 -X stuff 'python3.8 METh_BERT_entry.py -n 500 -sd 42 -s 4 -t MEBERT500\n'

screen -dmS METhSBERT_R
screen -S METhSBERT_R -p 0 -X stuff 'python3.8 METh_SBERT_entry.py -n 2000 -sd 42 -s 4 -t MESBERT2000\n'

screen -dmS MEOrgBERT_R
screen -S MEOrgBERT_R -p 0 -X stuff 'python3.8 MEOrg_BERT_entry.py -n 500 -sd 42 -s 4 -t MEOrgBERT500\n'

screen -dmS MEOrgSBERT_R
screen -S MEOrgSBERT_R -p 0 -X stuff 'python3.8 MEOrg_SBERT_entry.py -n 2000 -sd 42 -s 4 -t MEOrgSBERT2000\n'

screen -dmS BLEURT_R
screen -S BLEURT_R -p 0 -X stuff 'python3.8 BLEURT_entry.py -n 2000 -sd 42 -s 4 -t BLEURT2000\n'

screen -dmS BLEURT_REC_R
screen -S BLEURT_REC_R -p 0 -X stuff 'python3.8 BLEURT_REC_entry.py -n 500 -sd 42 -s 4 -t BLEURT_REC_500\n'

screen -dmS BLEU_R
screen -S BLEU_R -p 0 -X stuff 'python3.8 BLEU_entry.py -n 2000 -sd 42 -s 4 -t BLEU2000\n'

screen -dmS BERTScore_R
screen -S BERTScore_R -p 0 -X stuff 'python3.8 BERTScore_entry.py -n 2000 -sd 42 -s 4 -t BERTScore2000\n'

screen -dmS BERTScoreIDF_R
screen -S BERTScoreIDF_R -p 0 -X stuff 'python3.8 BERTScore_idf_entry.py -n 2000 -sd 42 -s 4 -t BERTScoreIDF2000\n'