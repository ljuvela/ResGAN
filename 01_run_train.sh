#!/bin/bash
# important: train on generated acoustic features

#python train_pls_noisegan.py --mode train --data ./traindata_genac/ 2>&1 > log_train.log

TRAINDATA=/home/ljuvela/CODE/pls_model/traindata_genac
#python train_pls_noisegan.py --mode train_noise_model --data $TRAINDATA --rnn_context_len=40 2>&1 > log_train_gan.log
python train_peek_fft_resgan.py --mode train --data ./traindata --rnn_context_len=40 --batch_size=128 2>&1 > log_train_gan.log
#python train_pls_noisegan.py --mode train --data $TRAINDATA --rnn_context_len=40 2>&1 > log_train.log

# only train GAN part
#python train_pls_noisegan.py --mode train_noise_model --data ./traindata_genac/ 2>&1 > log_train_gan.log

# generated acoustics and re-estimated pulses
#python train_pls_noisegan.py --mode train --data ./data_ac2glot_re/ 2>&1 > log_train_gan2.log
