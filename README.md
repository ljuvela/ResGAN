
This repo contains code for my ICASSP 2018 submission. I'll upload the neural net part once the review is complete. Residual generative adversarial networks are involved (hence the name ResGAN).

Meanwhile, some audio samples are available at http://tts.org.aalto.fi/mfcc_synthesis/.

The signal processing part contains nothing new, so thats also available.

## MFCC to AR envelope
The file `get_mfcc.py` contains code for computing MFCCs, pseudoinverting them to a magnitude spectrum, and fitting an all-pole model to the reconstructed spectrum. Basic signal processing stuff. 

Most of the code is in `numpy`, while `scipy` is used for solving the Toeplitz normal equations. `librosa` is required for the MFCCs.