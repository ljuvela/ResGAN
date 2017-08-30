#!/bin/bash
TESTDATA=/home/ljuvela/CODE/pls_model/testdata
#python train_pls_noisegan.py --mode generate --output ./output --testdata $TESTDATA --rnn_context_len=128

python train_peek_resgan.py --mode generate --output ./output --testdata $TESTDATA --rnn_context_len=40 --gan_model=./models/noise_gen_epoch19.model
