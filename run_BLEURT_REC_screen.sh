screen -dmS BLEURT_REC_R
screen -S BLEURT_REC_R -p 0 -X stuff 'python3.8 BLEURT_REC_entry.py -n 2000 -sd 42 -s 4 -t BLEURT_REC_2000\n'