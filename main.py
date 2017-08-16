import os
import numpy as np
# For reproducibility 
np.random.seed(9999)

import sys
sys.setrecursionlimit(10000)

# theano imports 
import theano
import theano.tensor as T
from theano.tensor import fft

from keras import backend as K

# keras imports
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input
from keras.layers.merge import  add, concatenate
from keras.layers.core import Activation, Flatten, Dropout, Lambda, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed 
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers.convolutional import UpSampling1D, Convolution1D, MaxPooling1D, AveragePooling1D, ZeroPadding1D
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

# 'Globals'
NFFT = 512
fbins = NFFT/2+1

max_voiced_freq = 1500
fs = 16000

# FFT analysis window
win_np = np.kaiser(400, 6.0)
win = theano.shared(win_np.astype(np.float32))

from data_utils import *

def theano_fft(x):

    # window with analysis window 
    x_win = win * x

    # zero-pad
    frame = T.zeros((x.shape[0], NFFT))
    frame = T.set_subtensor(frame[:,:x.shape[1]], x_win)
    
    # apply FFT
    x = fft.rfft(frame, norm='ortho')

    # get first half of spectrum
    x = x[:,:fbins] 
    # squared magnitude
    x = x[:,:, 0]**2 + x[:,:, 1]**2 

    # floor (prevents log from going to -Inf)
    x = T.maximum(x, 1e-9)

    return x 


# Keras wrapper for FFT layer
def fft_output_shape(x_shape):
    return (x_shape[0],fbins)

fft_layer = Lambda(theano_fft, output_shape=fft_output_shape)
fft_layer.trainable = False

# Keras wrapper for log 
def identity_output_shape(x_shape):
    return x_shape

def log_operation(x):
    return 10*T.log10(x)

log_layer = Lambda(log_operation, output_shape=identity_output_shape)
log_layer.trainable = False

def exp_operation(x):
    return T.pow(10.0, x/10.0)

exp_layer = Lambda(exp_operation, output_shape=identity_output_shape)
exp_layer.trainable = False

def win_operation(x):
    return win32_gpu*x

win_layer = Lambda(win_operation, output_shape=identity_output_shape)
win_layer.trainable = False

# fft model for transforming training set samples
def fft_model(model_name="fft_model"):
    x = Input(shape=(400,), name="fft_input")
    x_fft = fft_layer(x)
    #x_fft = log_layer(x_fft)
    model = Model(input=[x], output=[x_fft], name=model_name)
    return model

def glot_spec_model(timesteps=128, input_dim=48, output_dim=400, model_name="glot_spec_model"):

    ac_input = Input(shape=(timesteps, input_dim), name="ac_input")
 
    x = ac_input
    
    x = GRU(50, activation='relu', kernel_initializer='glorot_normal', 
              return_sequences=False, unroll=False)(x)
    
    x = Dense(output_dim)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Reshape((output_dim, 1))(x)    

    x = Convolution1D(filters=100,
                        kernel_size=15,
                        padding='same',
                        strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = Convolution1D(filters=100,
                      kernel_size=15,
                      padding='same',
                      strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)


    x = Convolution1D(filters=1,
                      kernel_size=15,
                      padding='same',
                      strides=1)(x)

    # remove singleton outer dimension 
    x = Reshape((output_dim,))(x)
        
    model = Model(input=[ac_input], output=[x], name=model_name)

    return model


def plot_feats(generated_feats, epoch, index, ext=''):
    plt.figure()
    for row in generated_feats:
        plt.plot(row)
    plt.savefig('figures/sample_spectra_epoch{}_index{}'.format(epoch, index) + ext + '.png')
    plt.close()

def train_model(BATCH_SIZE, data_dir, file_list, context_len=32, max_files=30):
    
    # History length for recurrent net
    timesteps = context_len

    optim = adam(lr=0.0001)
    model = glot_spec_model(timesteps=timesteps, input_dim=48, output_dim=fbins)
    model.compile(loss=['mse'], loss_weights=[1.0], optimizer=optim) 

    fft_mod = fft_model()

    # train glot model in time domain first 
    no_epochs = 40
    max_epochs_no_improvement = 5


    patience = max_epochs_no_improvement
    best_val_loss = 1e20
    for epoch in range(no_epochs):
        print("Training epoch is", epoch)
        epoch_error = 0.0
        total_batches = 0
        val_data = []
        for data in nc_data_provider(file_list, data_dir,
                                     max_files=max_files, context_len=timesteps):

            if len(val_data) == 0:
                val_data = data
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

                # simple log magnitude here, maybe do something smarter?
                x_feats_batch_fft = fft_mod.predict(x_feats_batch)
                x_feats_batch_fft = 10*np.log10(x_feats_batch_fft)

                d = model.train_on_batch([y_feats_batch],
                                         [x_feats_batch_fft])

                #import ipdb; ipdb.set_trace()
    
                    
                print("training batch %d, loss: %f" %
                      (index+total_batches, d))

                epoch_error += d

                if (index + total_batches) % 200 == 0:

                    spec = model.predict([y_feats_batch])
                    spec_gen = spec[0,:]
                    spec_ref = x_feats_batch_fft[0,:]
                    specs = np.array([spec_ref, spec_gen])
                    plot_feats(specs, epoch, index+total_batches, ext='.spec')
                    
            total_batches += no_batches

        epoch_error /= total_batches
    
        val_spec = fft_mod.predict(val_data[0])
        val_spec = 10.0*np.log10(val_spec)
        val_loss = model.evaluate([val_data[1]],
                                  [val_spec],
                                  batch_size=BATCH_SIZE)
        
        print("epoch %d validation loss: %f \n" %
              (epoch, val_loss))

        print("epoch %d training loss: %f \n" %
              (epoch, epoch_error))


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_epochs_no_improvement
            print ("New best model at epoch %d" % (epoch))
            model.save_weights('./pulse_spec.model')
        else:
            patience -= 1

        if patience == 0:
            break

    print "Finished training" 

def generate(file_list, data_dir, output_dir, context_len=32, max_files=1):
    
    model = glot_spec_model(timesteps=context_len, input_dim=48, output_dim=fbins)

    model.compile(loss='mse', optimizer="adam")
    
    model.load_weights('./pulse_spec.model')

    for data in nc_data_provider(file_list, data_dir, input_only=True,
                                 context_len=context_len, max_files=max_files):
        
        for fname, ac_data in data.iteritems():
            print fname
            spec_pred = model.predict([ac_data])
            out_file = os.path.join(args.output_dir, fname + '.spec')
            spec_pred.astype(np.float32).tofile(out_file)

           
    
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
    parser.add_argument("--rnn_context_len", type=int, default=32)
    parser.add_argument("--max_files", type=int, default=100)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":

        file_list = os.listdir(args.data_dir)

        train_model(BATCH_SIZE=args.batch_size, data_dir=args.data_dir,
                        file_list=file_list, max_files=args.max_files,
                        context_len=args.rnn_context_len)

    elif args.mode == "generate":
 
        test_dir = args.testdata_dir
        file_list = os.listdir(test_dir)

        generate(data_dir=test_dir, file_list=file_list,
                 output_dir=args.output_dir,
                 max_files=args.max_files,
                 context_len=args.rnn_context_len)

    
