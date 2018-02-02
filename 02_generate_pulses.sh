#!/bin/bash
python train_resgan.py --mode generate --output ./output --testdata ./testdata --rnn_context_len=40 --gan_model=./models/noise_gen_epoch19.model
