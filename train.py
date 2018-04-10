import os
import numpy as np

# For reproducibility 
np.random.seed(42)

# force cuda device (empty for CPU)
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import sys

# theano imports 
import theano
import theano.tensor as T
from theano.tensor import fft

from keras import backend as K

# keras imports
from keras.models import Model
from keras.layers import Dense, Reshape, Input
from keras.layers.merge import  add, concatenate, multiply
from keras.layers.core import Activation, Lambda
from keras.layers.recurrent import GRU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD, adam

import argparse
import math

# sklearn imports
from sklearn import preprocessing
from sklearn.externals import joblib

# netcdf for reading packaged data
from scipy.io import netcdf

import matplotlib as mpl
mpl.use('Agg') # no need for X-server
from matplotlib import pyplot as plt

from models import fft_model, time_glot_model, discriminator, generator, gan_container
from data_utils import nc_data_provider, norm_stats

# edge smoothing window
gen_filtwidths = np.asarray([15, 15, 15])
edgelen = sum(gen_filtwidths-1)
hannwin = np.hanning(edgelen)
smoothwin = np.concatenate((hannwin[:edgelen/2], np.ones(400-edgelen), hannwin[edgelen/2:]))

def plot_feats(generated_feats, epoch, index, ext='', fig_dir="./figures", fig_type=""):
    plt.figure()
    for row in generated_feats:
        plt.plot(row)
    plt.savefig(fig_dir + '/' + fig_type +'_epoch{}_index{}'.format(epoch, index) + ext + '.png')
    plt.close()

def train_pls_model(BATCH_SIZE, data_dir, file_list, context_len=32, max_files=30):
    
    no_epochs = 20
    max_epochs_no_improvement = 5

    timesteps = context_len

    optim = adam(lr=0.0001)
    pls_model = time_glot_model(timesteps=timesteps)
    pls_model.compile(loss=['mse', 'mse'], loss_weights=[1.0, 0.0], optimizer=optim) # disregard fft loss

    fft_mod = fft_model()

    patience = max_epochs_no_improvement
    best_val_loss = 1e20
    for epoch in range(no_epochs):
        print("Pre-train epoch is", epoch)
        epoch_error = [0.0, 0.0]
        total_batches = 0
        val_data = []
        for data in nc_data_provider(file_list, data_dir,
                                     max_files=max_files, context_len=timesteps):

            if len(val_data) == 0:
                val_data = data
                print("using data subset for validation")
                continue

            X_train = data[0]
            Y_train = data[1]

            no_batches = int(X_train.shape[0] / BATCH_SIZE)                
            print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

            # shuffle data
            ind = np.random.permutation(X_train.shape[0])
            X_train = X_train[ind]
            Y_train = Y_train[ind]
            for index in range(int(X_train.shape[0] / BATCH_SIZE)):
                x_feats_batch = X_train[
                    index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                y_feats_batch = Y_train[
                    index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

                x_feats_batch_fft = fft_mod.predict(x_feats_batch)
                
                d = pls_model.train_on_batch([y_feats_batch],
                                             [x_feats_batch, x_feats_batch_fft])
                    
                epoch_error += d

                if  (index + total_batches) % 500 == 0:

                    print("pre-training batch %d, wave loss: %f, spec loss %f" %
                          (index+total_batches, d[0], d[1]))

                    wave, spec = pls_model.predict([y_feats_batch])

                    wav_gen = wave[0,:]
                    wav_ref = x_feats_batch[0,:]
                    wavs = np.array([wav_ref, wav_gen])
                    plot_feats(wavs, epoch, index+total_batches, ext='.wave-pls')

                    spec_gen = spec[0,:]
                    spec_ref = x_feats_batch_fft[0,:]
                    specs = np.array([spec_ref, spec_gen])
                    plot_feats(specs, epoch, index+total_batches, ext='.spec-pls')
                    
            total_batches += no_batches                

        epoch_error[0] /= total_batches
        epoch_error[1] /= total_batches

        val_spec = fft_mod.predict(val_data[0])
        val_loss = pls_model.evaluate([val_data[1]],
                                      [val_data[0], val_spec],
                                      batch_size=BATCH_SIZE)
        
        print("epoch %d validation wave loss: %f ,spec loss %f \n" %
              (epoch, val_loss[0], val_loss[1]))

        print("epoch %d training wave loss: %f, spec loss %f \n" %
              (epoch, epoch_error[0], epoch_error[1]))
        
        # only on wave loss
        if val_loss[0] < best_val_loss:
            best_val_loss = val_loss[0]
            patience = max_epochs_no_improvement
            pls_model.save_weights('./pls.model')
        else:
            patience -= 1

        if patience == 0:
            break

    print "Finished training" 


def train_noise_model(BATCH_SIZE, data_dir, file_list, save_weights=False,
                      context_len=32, max_files=30, stats=None):

    no_epochs = 15
    
    timesteps = context_len

    optim_container = adam(lr=1e-4)
    optim_discriminator = SGD(lr=1e-5)

    fft_mod = fft_model()
    pls_model = time_glot_model(timesteps=timesteps)

    pls_model.compile(loss=['mse','mse'], loss_weights=[1.0, 1.0], optimizer='adam')
    pls_model.load_weights("./pls.model")

    disc_model = discriminator()
    gen_model = generator()
    disc_on_gen = gan_container(gen_model, disc_model)
 
    gen_model.compile(loss='mse', optimizer="adam")

    # use peek adversarial and peek mse loss for training generator
    disc_model.trainable = False
    disc_on_gen.compile(loss=['mse','mse'], loss_weights=[1.0, 1.0], optimizer=optim_container) 

    # don't use peek loss for discriminator
    disc_model.trainable = True
    disc_model.compile(loss=['mse','mse'], loss_weights=[1.0, 0.0], optimizer=optim_discriminator) 

    print "Discriminator model:"
    print disc_model.summary()
    print "Generator model:"
    print gen_model.summary()
    print "Joint model:"
    print disc_on_gen.summary()

    label_fake = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
    label_real = np.ones((BATCH_SIZE, 1), dtype=np.float32)

    # train residual GAN with FFT     
    for epoch in range(no_epochs):
        print("Epoch is", epoch)

        epoch_error = 0
        total_batches = 0
      
        for data in nc_data_provider(file_list, data_dir,
                                     max_files=max_files, context_len=timesteps):   

            X_train = data[0]
            Y_train = data[1]

            pls_len = X_train.shape[1]

            no_batches = int(X_train.shape[0] / BATCH_SIZE)    

            # shuffle data
            ind = np.random.permutation(X_train.shape[0])
            X_train = X_train[ind]
            Y_train = Y_train[ind]
            for index in range(int(X_train.shape[0] / BATCH_SIZE)):
                x_feats_batch = X_train[
                    index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                y_feats_batch = Y_train[
                    index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

                x_pred_batch, x_pred_batch_fft = pls_model.predict([y_feats_batch])
                                
                pls_pred = x_pred_batch
                pls_real = x_feats_batch

                # smoothing windows to prevent edge effects
                pls_pred *= smoothwin
                pls_real *= smoothwin

                # evaluate target fft
                fft_real = fft_mod.predict(pls_real)

                noise = np.random.randn(BATCH_SIZE, pls_len)

                # train generator through discriminator
                _, peek_real = disc_model.predict([pls_real, fft_real])
                disc_model.trainable = False
                loss_g = disc_on_gen.train_on_batch([pls_pred, noise], [label_real, peek_real])
 
                noise = np.random.randn(BATCH_SIZE, pls_len)

                # train discriminator with real data
                disc_model.trainable = True
                loss_dr = disc_model.train_on_batch([pls_real, fft_real], [label_real, peek_real])

                # train discriminator with fake data
                pls_fake, fft_fake = gen_model.predict([pls_pred, noise])
                loss_df = disc_model.train_on_batch([pls_fake, fft_fake], [label_fake, peek_real])
        
                if (index + total_batches) % 500 == 0:

                    print("training batch %d, G loss: %f, D loss (real): %f, D loss (fake): %f" %
                          (index + total_batches, loss_g[0], loss_dr[0], loss_df[0]))

                if (index + total_batches) % 500 == 0:

                    wav_ref = pls_real[0,:]
                    wav_gen = pls_pred[0,:]
                    wav_noised = pls_fake[0,:]
                    wavs = np.array([wav_ref, wav_gen, wav_noised])
                    plot_feats(wavs, epoch, index+total_batches, ext='.wave')
                 
            total_batches += no_batches

        gen_model.save_weights('./models/noise_gen_epoch' + str(epoch) + '.model')

    print "Finished noise model training" 

def generate(file_list, data_dir, output_dir, context_len=32, stats=None,
             base_model_path='./pls.model', gan_model_path='./noise_gen.model'):
    
    pulse_model = time_glot_model(timesteps=context_len)
    gan_model = generator()
    
    pulse_model.compile(loss='mse', optimizer="adam")
    gan_model.compile(loss='mse', optimizer="adam")

    pulse_model.load_weights(base_model_path)
    gan_model.load_weights(gan_model_path)

    for data in nc_data_provider(file_list, data_dir, input_only=True, 
                                 context_len=context_len):
        for fname, ac_data in data.iteritems():
            print fname
                                              
            pls_pred, _ = pulse_model.predict([ac_data])
            noise = np.random.randn(pls_pred.shape[0], pls_pred.shape[1])
            pls_gan, _ = gan_model.predict([pls_pred, noise])
            
            out_file = os.path.join(args.output_dir, fname + '.pls')
            pls_gan.astype(np.float32).tofile(out_file)

            out_file = os.path.join(args.output_dir, fname + '.pls_nonoise')
            pls_pred.astype(np.float32).tofile(out_file)
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str,
                        default="./traindata")
    parser.add_argument("--testdata_dir", type=str,
                        default="./testdata")
    parser.add_argument("--output_dir", type=str,
                        default="./output")
    parser.add_argument("--rnn_context_len", type=int, default=64)
    parser.add_argument("--max_files", type=int, default=100)
    parser.set_defaults(nice=False)
    parser.add_argument("--gan_model", type=str,
                        default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":

        file_list = os.listdir(args.data_dir)

        train_pls_model(BATCH_SIZE=args.batch_size, data_dir=args.data_dir,
                        file_list=file_list, max_files=args.max_files,
                        context_len=args.rnn_context_len)

        stats = norm_stats(file_list[0], args.data_dir)

        train_noise_model(BATCH_SIZE=args.batch_size, data_dir=args.data_dir,
                          file_list=file_list, max_files=args.max_files,
                          context_len=args.rnn_context_len,
                          stats=stats)


    elif args.mode == "train_pulse_model":
        print ("MODE: Training time domain pulse model")
    
        file_list = os.listdir(args.data_dir)

        train_pls_model(BATCH_SIZE=args.batch_size, data_dir=args.data_dir,
                        file_list=file_list, max_files=args.max_files,
                        context_len=args.rnn_context_len)
        
    elif args.mode == "train_noise_model":
        print ("MODE: Training noise model")
    
        file_list = os.listdir(args.data_dir)

        stats = norm_stats(file_list[0], args.data_dir)

        train_noise_model(BATCH_SIZE=args.batch_size, data_dir=args.data_dir,
                          file_list=file_list, max_files=args.max_files,
                          context_len=args.rnn_context_len,
                          stats=stats)

    elif args.mode == "generate":
 
        test_dir = args.testdata_dir
        file_list = os.listdir(test_dir)

        stats = norm_stats(file_list[0], test_dir)

        generate(data_dir=test_dir, file_list=file_list,
                 output_dir=args.output_dir,
                 context_len=args.rnn_context_len, stats=stats,
                 gan_model_path=args.gan_model)

        
