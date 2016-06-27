#!/usr/bin/env python
#
# Convert wav files and build TF records for
# training.
#
# $ pip install tqdm
# $ source tf/bin/activate
# $ ./tfbuild.py

import tensorflow as tf
import numpy as np
import scipy, numpy as np
from scipy.io import wavfile
import glob
import math
from os import path
import re
import random
import sys
from tqdm import tqdm

DIR = './data/wav'
N = 256  # FFT size
IMG_X = 200  # this can be changed (max 229 for .330ms sample)
IMG_Y = (N/2) + 1
NUM_INPUTS = IMG_X * IMG_Y # Total number of pixels

TRAINING_FILE = "./data/singles.training.tfrecords"
VALIDATION_FILE = "./data/singles.validation.tfrecords"
TEST_FILE = "./data/singles.test.tfrecords"

# Returns a 2D array of STFT frames. Each frame represents a time duration
# of N samples, and consists of N/2 complex numbers, one per frequency bin.
#
# The overlap represents the amount of time overlap between frames. So a value
# of 4 means 25% overlapping frames.
def stft(x, fftsize=2048, overlap=4):   
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

# Normalize np.array to standard (0-mean, x stddev)
def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

# One-hot encode values in batch.
def one_hot(batch, max_value):
    b = np.zeros((len(batch), max_value), dtype='int32')
    b[np.arange(len(batch)), batch] = 1
    return b

# Shuffle iterator
def shuffle(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

def load_audio_data(file, N=N):
    # Audio data format:
    #
    # $ sox --i data/wav/singles-V100-P56-attack.wav
    # 
    #   Input File     : 'data/wav/singles-V100-P56-attack.wav'
    #   Channels       : 1
    #   Sample Rate    : 44100
    #   Precision      : 16-bit
    #   Duration       : 00:00:00.33 = 14553 samples = 24.75 CDDA sectors
    #   File Size      : 29.2k
    #   Bit Rate       : 707k
    #   Sample Encoding: 16-bit Signed Integer PCM
    #
    # Each sample is a 16-bit signed integer (PCM = linear)
    rate, audio_data = wavfile.read(file)
    sy = stft(audio_data, N, 4)   # sy.shape = [N/2, duration/hops]
    sgram = (2.0 / N) *scipy.absolute(sy.T) # calculate spectrogram
    nsgram = normalize(sgram)
    return nsgram.flatten()[:NUM_INPUTS]

def get_note_label(file):
    name = path.basename(file)
    match = re.match(r'.*-N(\d+)-', name)
    if match != None:
        note = match.groups()[0]
        return int(note)
    else:
        raise ValueError("Unlabled file: " + file)
            
def get_examples(dir):
    return glob.iglob(dir+'/*.wav')

def write_examples(examples, file):
    print "Writing:", file
    writer = tf.python_io.TFRecordWriter(file)
    # iterate over each example
    # wrap with tqdm for a progress bar
    for example in tqdm(examples):
        filename = example
        features = load_audio_data(example)
        label = get_note_label(example)

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])),
                'sample': tf.train.Feature(
                    float_list=tf.train.FloatList(value=features.astype("float"))),
                'filename': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=example)),
        }))
    
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)


# Get list of examples
examples = list(shuffle(get_examples(DIR)))

# break up: 60% training, 20% validation, 20% test
total = len(examples)
p20 = int(math.floor((20.0 / 100) * total))
tr_idx = p20 * 3
va_idx = tr_idx + p20

print "Split (training/validation/test):", tr_idx, p20, p20
write_examples(examples[:tr_idx], TRAINING_FILE)
write_examples(examples[tr_idx+1:va_idx], VALIDATION_FILE)
write_examples(examples[va_idx+1:], TEST_FILE)



