
screen -dmS BERTScore
screen -S BERTScore -p 0 -X stuff 'python3.8 BERTScore_entry.py -n 2000 -sd 42 -s 4 -t BERTScore2000\n'

screen -dmS BERTScoreIDF
screen -S BERTScoreIDF -p 0 -X stuff 'python3.8 BERTScore_idf_entry.py -n 2000 -sd 42 -s 4 -t BERTScoreIDF2000\n'