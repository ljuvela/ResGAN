#!/bin/bash
#python train_peek_fft_resgan.py --mode train --data ./traindata --rnn_context_len=40 --batch_size=128
python train_peek_fft_resgan.py --mode train_noise_model --data ./traindata --rnn_context_len=40 --batch_size=128
