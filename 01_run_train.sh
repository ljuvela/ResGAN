#!/bin/bash
python train.py --mode train --data_dir=./traindata --rnn_context_len=1 --batch_size=128
