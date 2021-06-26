
screen -dmS BLEURT
screen -S BLEURT -p 0 -X stuff 'python3.8 BLEURT_entry.py -n 2000 -sd 42 -s 4 -t BLEURT2000\n'