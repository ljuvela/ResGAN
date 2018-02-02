#!/bin/bash
python train_resgan.py --mode train --data ./traindata --rnn_context_len=40 --batch_size=128
