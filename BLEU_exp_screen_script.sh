
screen -dmS BLEU
screen -S BLEU -p 0 -X stuff 'python3.8 BLEU_entry.py -n 2000 -sd 42 -s 4 -t BLEU2000\n'