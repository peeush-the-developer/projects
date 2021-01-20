#!/bin/sh

MODEL_NAME=$1 #Argument from CLI passed to run the shell script

python3 train.py --fold 0 --model $MODEL_NAME
python3 train.py --fold 1 --model $MODEL_NAME
python3 train.py --fold 2 --model $MODEL_NAME
python3 train.py --fold 3 --model $MODEL_NAME
python3 train.py --fold 4 --model $MODEL_NAME