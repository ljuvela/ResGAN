import sys
import argparse

import numpy as np
from numpy.linalg import pinv

from librosa.core.time_frequency import fft_frequencies, mel_frequencies
import librosa

import  scipy.linalg as LA  
from scipy.signal import freqz, convolve, deconvolve, lfilter


def lsf2poly(L):

    # always use double precision 
    dtype = L.dtype
    L = L.astype(np.float64)

    order = len(L)
    Q = L[::2]
    P = L[1::2]
    poles_P = np.r_[np.exp(1j*P),np.exp(-1j*P)]
    poles_Q = np.r_[np.exp(1j*Q),np.exp(-1j*Q)]
    
    P = np.poly(poles_P)
    Q = np.poly(poles_Q)
    
    # convolve from scipy.signal
    # only supports even orders
    P = convolve(P, np.array([1.0, -1.0]))
    Q = convolve(Q, np.array([1.0, 1.0]))
    
    a = 0.5*(P+Q)
 
    a = a[:-1]

    return a.astype(dtype) 

def poly2lsf(a):
    a = a / a[0]        
    A = np.r_[a, 0.0]
    B = A[::-1]
    P = A - B  
    Q = A + B  
    
    P = deconvolve(P, np.array([1.0, -1.0]))[0]
    Q = deconvolve(Q, np.array([1.0, 1.0]))[0]
    
    roots_P = np.roots(P)
    roots_Q = np.roots(Q)
    
    angles_P = np.angle(roots_P[::2])
    angles_Q = np.angle(roots_Q[::2])
    angles_P[angles_P < 0.0] += np.pi
    angles_Q[angles_Q < 0.0] += np.pi
    lsf = np.sort(np.r_[angles_P, angles_Q])
    return lsf

# mel filterbank function modified from librosa
def get_filterbank(n_filters=60, NFFT=512,  fs=16000, fmin=0.0, fmax=None,
                   htk=False, normalize=False):


    n_mels = n_filters

    if fmax is None:
        fmax = float(fs) / 2

    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + NFFT // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=fs, n_fft=NFFT)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    # to make evenly spaced filterbank, use fft_frequencies

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if normalize == True:
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    return weights

def lsf2mfbe(lsf, mel_filters):

    NFFT = 512
    M = get_filterbank(n_filters=mel_filters, NFFT=NFFT, normalize=False, htk=True)

    mfbe = np.zeros(( len(lsf), mel_filters), dtype=np.float64)
    spec = np.zeros((len(lsf), NFFT/2+1,), dtype=np.float64)
   
    x = np.zeros((NFFT,), dtype=np.float64)
    x[0] = 1.0
    b = np.ones((1,), dtype=np.float64)
 
    for i, lsf_vec in enumerate(lsf):     
        #convert lsf to filter polynomial
        a_poly = lsf2poly(lsf_vec)
        # compute power spectrum
        w, H = freqz(b=1.0, a=a_poly, worN=NFFT, whole=True)
        spec_vec = np.abs(H[:(NFFT/2+1)])
        #spec_vec = np.square(spec_vec)
        # apply filterbank matrix
        mfbe[i,:] = np.log10( np.dot(M,spec_vec) )
        spec[i,:] = spec_vec
         
    return mfbe, spec

def mfbe2lsf(mfbe, lsf_order):

    NFFT = 512
    M = get_filterbank(n_filters=mfbe.shape[1], NFFT=NFFT, normalize=False, htk=True)

    M_inv = pinv(M)
    p = lsf_order

    lsf = np.zeros(( len(mfbe), lsf_order), dtype=np.float64)
    spec = np.zeros((len(mfbe), NFFT/2+1), dtype=np.float64)

    for i, mfbe_vec in enumerate(mfbe):
    
        # invert mel filterbank
        spec_vec = np.dot(M_inv, np.power(10, mfbe_vec))

        # floor reconstructed spectrum
        spec_vec = np.maximum(spec_vec, 1e-9)
 
        # squared magnitude 2-sided spectrum
        twoside = np.r_[spec_vec, np.flipud(spec_vec[1:-1])]
        twoside = np.square(twoside) 
        r = np.fft.ifft(twoside)
        r = r.real

        # reference from talkbox
        # a,_,_ = TB.levinson(r, order=p)
  
        # levinson-durbin
        a = LA.solve_toeplitz(r[0:p],r[1:p+1])
        a = np.r_[1.0, -1.0*a]
   
        lsf[i,:] = poly2lsf(a)
   
        # reconstructed all-pole spectrum
        w, H = freqz(b=1.0, a=a, worN=NFFT, whole=True)
        spec[i,:] = np.abs(H[:(NFFT/2+1)])
            
    return lsf, spec

def spec2lsf(spec, lsf_order=30):

    NFFT = 2*(spec.shape[0]-1)
    n_frames = spec.shape[1]

    p = lsf_order

    lsf = np.zeros(( n_frames, lsf_order), dtype=np.float64)
    spec_rec = np.zeros(spec.shape)

    for i, spec_vec in enumerate(spec.T):
    
        # floor reconstructed spectrum
        spec_vec = np.maximum(spec_vec, 1e-9)
 
        # squared magnitude 2-sided spectrum
        twoside = np.r_[spec_vec, np.flipud(spec_vec[1:-1])]
        twoside = np.square(twoside) 
        r = np.fft.ifft(twoside)
        r = r.real
  
        # levinson-durbin
        a = LA.solve_toeplitz(r[0:p],r[1:p+1])
        a = np.r_[1.0, -1.0*a]
   
        lsf[i,:] = poly2lsf(a)
   
        # reconstructed all-pole spectrum
        w, H = freqz(b=1.0, a=a, worN=NFFT, whole=True)
        spec_rec[:,i] = np.abs(H[:(NFFT/2+1)])
            
    return lsf, spec_rec
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lsf_file", type=str, default=None)
    parser.add_argument("--mfcc_file", type=str, default=None)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--win_length", type=int, default=480)
    parser.add_argument("--hop_size", type=int, default=80)
    parser.add_argument("--lsf_order", type=int, default=30)
    parser.add_argument("--mel_filters", type=int, default=24)
    parser.add_argument("--mfcc_order", type=int, default=20)
    parser.add_argument("--nfft", type=int, default=1024)

    args = parser.parse_args()
    return args

# read single precision float binary file (HTS format)
def read_binary_file(fname, order):
    data = np.fromfile(fname, dtype=np.float32)
    return data.reshape(-1, order)

def get_mfcc_lsf(params):

    sig, _ = librosa.load(params.input_file, sr=params.sample_rate, mono=True)

    n_frames = int(np.ceil(len(sig) / (1.0 * params.hop_size)))
    spec_pow = 1.0

    # pre-emphasis
    b = np.asarray([1.0 , -.97])
    a = np.asarray([1.0])
    sig = lfilter(b, a, sig)
    
    # STFT
    fbins = params.nfft/2 + 1
    #spec=np.zeros((fbins, n_frames), dtype=np.float32)
    librosa_spec = librosa.core.stft(sig, n_fft=params.nfft, win_length=params.win_length,
                          center=True, hop_length=params.hop_size)
    #spec[:,:n_frames] = librosa_spec[:,:n_frames]                          
    spec = librosa_spec[:,:n_frames]

    spec = np.abs(spec)
    # floor for zero frames
    spec = np.maximum(spec, 1e-9)
    energy = 10.0 * np.log10(np.sum(spec**2, axis=0))
    spec = spec**spec_pow

    # mel filterbank
    M = get_filterbank(n_filters=params.mel_filters, NFFT=params.nfft,  fs=params.sample_rate, fmin=0.0, fmax=None,
                   htk=True, normalize=True)

    mfbe = M.dot(spec)               

    # log 
    lmfbe = 20.0 / spec_pow * np.log10(mfbe)
    
    # DCT
    D = librosa.filters.dct(params.mfcc_order, params.mel_filters)
    mfcc = D.dot(lmfbe)
    mfcc[0,:] = energy

    # invert DCT
    Dinv = pinv(D)
    lmfbe_r = Dinv.dot(mfcc)
    
    # exp
    mfbe_r = np.power(10.0, lmfbe_r/20.0 * spec_pow)

    # invert mel filterbank
    Minv = pinv(M)
    spec_r = Minv.dot(mfbe_r)

    # clip negative values
    spec_r = np.maximum(spec_r, 1e-9)

    # get LSFs, takes amplitude spectrum (not squared)
    spec_r = spec_r ** (1.0 / spec_pow)
    lsf_r, spec_r_ar = spec2lsf(spec_r, lsf_order=params.lsf_order)

    # transpose mfcc to standard ordering 
    mfcc = mfcc.T

    return lsf_r, mfcc



if __name__ == "__main__":
    
    params = get_args()

    lsf, mfcc = get_mfcc_lsf(params)

    if params.lsf_file != None:
        lsf.astype(np.float32).tofile(params.lsf_file)

    if params.mfcc_file != None:
        mfcc.astype(np.float32).tofile(params.mfcc_file)
   
