#!/bin/bash
# important: train on generated acoustic features

#python train_pls_noisegan.py --mode train --data ./traindata_genac/ 2>&1 > log_train.log

TRAINDATA=/scratch/work/ljuvela/DATA/traindata_nick
python train_pls_noisegan.py --mode train_pls_model --data $TRAINDATA --max_files=2

# only train GAN part
#python train_pls_noisegan.py --mode train_noise_model --data ./traindata_genac/ 2>&1 > log_train_gan.log

# generated acoustics and re-estimated pulses
#python train_pls_noisegan.py --mode train --data ./data_ac2glot_re/ 2>&1 > log_train_gan2.log
