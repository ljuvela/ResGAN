import os
import numpy as np

# sklearn imports
from sklearn import preprocessing
from sklearn.externals import joblib

# netcdf for reading packaged data
from scipy.io import netcdf

# function for loading data.nc NetCDF files packaged with Xin's tools
def load_data(files_list, data_dir, num_files=30, context_len=32):
    #files_list = os.listdir(data_dir)
    output_data = None
    input_data = None
    tags = []
    seq_lengths = []

    for fname in files_list[:num_files]:
        print fname
        f = os.path.join(data_dir, fname)
        with netcdf.netcdf_file(f, 'r') as fid:

            seq_tags = fid.variables['seqTags'][:].copy()
            for t in seq_tags:
                tags.append(''.join(t))

            lens = fid.variables['seqLengths'][:].copy()
            for l in lens:
                seq_lengths.append(l)

            # remove column-wise normalization (required for conv-nets)
            m = fid.variables['outputMeans'][:].copy()
            s = fid.variables['outputStdevs'][:].copy()
            feats = fid.variables['targetPatterns'][:].copy()
            input_feats = fid.variables['inputs'][:].copy()
            scaler = preprocessing.StandardScaler()
            scaler.mean_ = m
            scaler.scale_ = s
            feats = scaler.inverse_transform(feats)
            assert feats.shape[0] == input_feats.shape[0]

        if output_data == None and input_data == None:
            output_data = feats
            input_data = input_feats
        else:
            input_data = np.vstack((input_data, input_feats))
            output_data = np.vstack((output_data, feats))

    # cast list to numpy array
    seq_lengths = np.asarray(seq_lengths)

    input_dim = input_data.shape[1]
    output_dim = output_data.shape[1]

    input_data_seqs = np.zeros((len(input_data), context_len, input_dim), dtype=np.float32)

    sample_ind = 0 # running index for sample in full dataset (batch dimension)                  
    start_ind = 0 # running sequence start index in concatenated data matrix 
    for seq_ind, seq_len in enumerate(seq_lengths):
        # prepend zero context frames to sequence                            
        padded_seq_data = np.vstack((np.zeros((context_len-1, input_dim), dtype=np.float32),
                                              input_data[start_ind:start_ind+seq_len, :]))

        for frame_index in range(seq_len):
            input_data_seqs[sample_ind,:,:] = padded_seq_data[frame_index:frame_index+context_len,:]
            sample_ind += 1

        start_ind += seq_len

    return output_data, input_data_seqs, tags, seq_lengths

# function for loading data.nc NetCDF files packaged with Xin's tools
def load_test_data(files_list, data_dir, num_files=30, context_len=32):

    input_data = None
    tags = []
    seq_lengths = []

    for fname in files_list[:num_files]:
        print fname
        f = os.path.join(data_dir, fname)
        with netcdf.netcdf_file(f, 'r') as fid:

            seq_tags = fid.variables['seqTags'][:].copy()
            for t in seq_tags:
                tags.append(''.join(t))

            lens = fid.variables['seqLengths'][:].copy()
            for l in lens:
                seq_lengths.append(l)


            input_feats = fid.variables['inputs'][:].copy()

        if  input_data == None:
            input_data = input_feats
        else:
            input_data = np.vstack((input_data, input_feats))


    # cast list to numpy array
    seq_lengths = np.asarray(seq_lengths)

    input_dim = input_data.shape[1]

    # initialize sequences
    input_sequences = dict.fromkeys(tags)

    start_ind = 0 # running sequence start index in concatenated data matrix
    for seq_ind, seq_len in enumerate(seq_lengths):

        input_data_seqs = np.zeros((seq_len, context_len, input_dim), dtype=np.float32)
        # prepend zero context frames to sequence
        padded_seq_data = np.vstack((np.zeros((context_len-1, input_dim), dtype=np.float32),
                                              input_data[start_ind:start_ind+seq_len, :]))

        for frame_index in range(seq_len):
            input_data_seqs[frame_index,:,:] = padded_seq_data[frame_index:frame_index+context_len,:]

        start_ind += seq_len

        # set input data sequence by tag                 
        input_sequences[tags[seq_ind]] = input_data_seqs

    return input_sequences

def read_binary_file(file, dim=1):
    f = open(file, 'rb')
    data = np.fromfile(f, dtype=np.float32)
    assert data.shape[0] % dim == 0.
    data = data.reshape(-1, dim)
    return data

class nc_data_provider:

    def __init__(self, file_list, data_dir, input_only=False, context_len=32, max_files=30):
        self.i = 0
        self.file_list = file_list
        self.data_dir = data_dir
        self.input_only = input_only
        self.context_len = context_len
        self.max_files = max_files

    def __iter__(self):
        return self

    def next(self):
        if (self.i >= len(self.file_list)) or (self.i >= self.max_files):
            raise StopIteration
        else:
            if self.input_only:
                in_data = load_test_data([self.file_list[self.i]], self.data_dir, 
                                         context_len=self.context_len)
                self.in_data = in_data
                self.i += 1
                return self.in_data
            else:
                out_data, in_data, _, _ = load_data([self.file_list[self.i]], self.data_dir,
                                                    context_len=self.context_len)
                self.out_data = out_data
                self.in_data = in_data
                self.i += 1
                return [self.out_data, self.in_data]
