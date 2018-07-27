import numpy as np
# For reproducibility 
np.random.seed(9999)

import sys
import tensorflow as tf
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

# 'Globals'
fbins = 400
NFFT = (fbins-1)*2


def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def tensorflow_fft(x, nfft=NFFT, log_scale=0.1):

     # clip for safety
    x = tf.clip_by_value(x, clip_value_min=-1e9, clip_value_max=1e9)

    # get shape parameters
    win_len = tf.shape(x)[-1]
    nfft = (win_len-1) * 2

    # fft with zero padding 
    pad = tf.zeros_like(x[:, :(nfft-win_len)])
    x_pad = tf.concat([x,pad], axis=-1)
    x_c =  tf.cast(x_pad, dtype=tf.complex64)
    X = tf.spectral.fft(x_c)
    X = X[:,:win_len]
    X = X / np.sqrt(1.0*NFFT)
    X = tf.abs(X)**2
    X = tf.cast(X, dtype=x.dtype) # back to original datatype!!

    # floor to prevent log of zero
    X = tf.maximum(X, 1e-9)
    # map to log domain where 0dB -> 1 and -90dB -> -1
    X = tf_log10(X) * (20.0/90.0) + 1.0
    # scale to weigh errors
    X = log_scale*X 
  
    return X

# Keras wrapper for FFT layer
def fft_output_shape(x_shape):
    return (x_shape[0],fbins)

#fft_layer = Lambda(theano_fft, output_shape=fft_output_shape)
fft_layer = Lambda(tensorflow_fft, output_shape=fft_output_shape)
fft_layer.trainable = False

# Keras wrapper for log 
def identity_output_shape(x_shape):
    return x_shape

def log_operation(x):
    return tf_log10(x)

log_layer = Lambda(log_operation, output_shape=identity_output_shape)
log_layer.trainable = False

def exp_operation(x):
    return tf.pow(10.0, x/10.0)

exp_layer = Lambda(exp_operation, output_shape=identity_output_shape)
exp_layer.trainable = False

# fft model for transforming training set samples
def fft_model(model_name="fft_model"):
    x = Input(shape=(400,), name="fft_input")
    x_fft = fft_layer(x)
    model = Model(inputs=[x], outputs=[x_fft], name=model_name)
    return model

def time_glot_model(timesteps=128, input_dim=22, output_dim=400, model_name="time_glot_model"):

    ac_input = Input(shape=(timesteps, input_dim), name="ac_input")
 
    x_t = ac_input
    
    x_t = GRU(50, activation='relu', kernel_initializer='glorot_normal', 
              return_sequences=False, unroll=False)(x_t)
    
    x = x_t
    
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

    x_t = x
        
    x_fft = fft_layer(x)

    model = Model(inputs=[ac_input], outputs=[x_t, x_fft], name=model_name)

    return model

def generator(input_dim=400, ac_dim=22, output_dim=400):
    
    pls_input = Input(shape=(input_dim,), name="pls_input")
    noise_input = Input(shape=(input_dim,), name="noise_input")

    pls = Reshape((input_dim, 1))(pls_input)    
    noise = Reshape((input_dim, 1))(noise_input)
 
    x = concatenate([pls, noise], axis=2) # concat as different channels

    x = Convolution1D(filters=100,
                        kernel_size=15,
                        padding='same',
                        strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = concatenate([pls, x], axis=2) # concat as different channels

    x = Convolution1D(filters=100,
                      kernel_size=15,
                      padding='same',
                      strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = concatenate([pls, x], axis=2) # concat as different channels

    x = Convolution1D(filters=100,
                        kernel_size=15,
                        padding='same',
                        strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = concatenate([pls, x], axis=2) # concat as different channels
                    
    x = Convolution1D(filters=1,
                      kernel_size=15,
                      padding='same',
                        strides=1)(x)

    x = Activation('tanh')(x)

    # force additivity   
    x = add([pls, x]) 
             
    # remove singleton outer dimension 
    x = Reshape((output_dim,))(x)

    # add fft channel to output
    x_fft = fft_layer(x)
     
    model = Model(inputs=[pls_input, noise_input], outputs=[x, x_fft],
                  name="generator")

    return model

def discriminator(input_dim=400):

    pls_input = Input(shape=(input_dim,), name="pls_input") 
    fft_input = Input(shape=(input_dim,), name="fft_input") 

    x = Reshape((input_dim, 1))(pls_input)
    x_fft = Reshape((input_dim, 1))(fft_input)    

    x = concatenate([x, x_fft], axis=2) # concat as different channels
    
    # input shape batch_size x 1 (number of channels) x 400 (length of pulse)
    x = Convolution1D(filters=64,
                        kernel_size=7,
                        strides=3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    # shape [batch_size x 64 x 132]
    x = Convolution1D(filters=128,
                        kernel_size=7,
                        strides=3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    # shape [batch_size x 128 x 42]
    x = Convolution1D(filters=256,
                        kernel_size=7,
                        strides=3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    peek_output = x # used for generator training regularization

    # shape [batch_size x 256 x 12]
    x = Convolution1D(filters=128,
                        kernel_size=5,
                        strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    # shape [batch_size x 128 x 4]                                                             
 
    #nn.Sigmoid() # use sigmoid for normal gan, commented out for LS-GAN                                 
    x = Convolution1D(filters=1,
                      kernel_size=3,
                      strides=2)(x)

    # shape [batch_size x 1 x 1] 
    x = Reshape((1,))(x)

    model = Model(inputs=[pls_input, fft_input], outputs=[x, peek_output],
                  name="discriminator")

    return model

def gan_container(generator, discriminator, input_dim=400, ac_dim=22):
   
    discriminator.trainable = False

    pls_input = Input(shape=(input_dim,), name="pls_input")
    noise_input = Input(shape=(input_dim,), name="noise_input")

    x, x_fft = generator([pls_input, noise_input])
    x, peek_output = discriminator([x, x_fft])

    model = Model(inputs=[pls_input, noise_input], outputs=[x, peek_output],
                  name="gan_container")
    return model

