
screen -dmS METhBERT
screen -S METhBERT -p 0 -X stuff 'python3.8 METhBERT_entry.py -n 100 -sd 42 -s 4 -t MEBERT100\n'

screen -dmS METhSBERT
screen -S METhSBERT -p 0 -X stuff 'python3.8 METhSBERT_entry.py -n 2000 -sd 42 -s 4 -t MESBERT2000\n'
