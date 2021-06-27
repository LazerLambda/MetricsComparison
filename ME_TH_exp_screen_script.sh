
screen -dmS MEBERT
screen -S MEBERT -p 0 -X stuff 'python3.8 MEBERT_entry.py -n 100 -sd 42 -s 4 -t MEBERT100\n'

screen -dmS MESBERT
screen -S MESBERT -p 0 -X stuff 'python3.8 MESBERT_entry.py -n 2000 -sd 42 -s 4 -t MESBERT2000\n'
