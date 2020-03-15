import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os
from keras.utils.np_utils import to_categorical

import helpers
from helpers import Config, DefaultConfig

def note_to_freq(note, octave, a_440=440.0):
    notes = ['A', 'As', 'B', 'C', 'Cs', 'D',
        'Ds', 'E', 'F', 'Fs', 'G', 'Gs']
    note_map = dict(zip(notes, range(len(notes))))
    octave = int(octave)

    key = note_map[note] + ((octave - 1) * 12) + 1

    freq = a_440 * pow(2, (key - 49) / 12.0)
    return freq

def gen_spectrogram(file, config=DefaultConfig):
    fs, data = wavfile.read(file)
    if config.resample > 0:
        number_of_samples = round(len(data) * float(config.resample) / fs)
        data = signal.resample(data, number_of_samples)
        fs = config.resample

    if config.no_spectrogram:
        Sxx = np.reshape(data, (config.rows, config.cols))
    else:
        f, t, Sxx = signal.spectrogram(data, fs,
                                        window=('hann'),
            nperseg=config.s_nperseg,
            nfft=config.s_nfft,
            noverlap=config.s_noverlap,
            mode='magnitude')

        if config.log_scale:
            np.log(Sxx, out=Sxx)

    return Sxx

def chord_parts(f):
    parts = f.split("-")
    partmap = {
        "note": parts[0],
        "patch": parts[1],
        "chord": parts[2],
        "inversion": parts[3],
        "envelope": parts[4],
    }
    partmap["root"] = re.match('([A-G]+[bs]?)(\d*)', parts[0]).group(1)
    return partmap

def note_parts(f):
    parts = f.split("-")
    partmap = {
        "note": parts[1],
        "patch": parts[2],
        "freq": float(parts[3]),
        "shift": parts[4],
        "volume": parts[5],
        "reject": parts[6],
        "envelope": parts[7],
    }

    matches = re.match('([A-G]+[bs]?)(\d*)', parts[1])
    partmap["root"] = matches.group(1)
    partmap["octave"] = matches.group(2)
    # partmap["freq"] = DeepSamples.note_to_freq(partmap["root"], partmap["octave"])
    return partmap

def make_categories(classes):
    categories = {}
    for i, ccls in enumerate(classes):
        categories[ccls] = to_categorical(i, num_classes=len(classes))
    return categories


class ChordSamples:
  def __init__(self, dir_name="./samples", num_samples=20000, split=0.8, config=DefaultConfig):
    self.config = config
    self.dir_name = dir_name  # path to samples
    self.num_samples = num_samples # number of samples to load
    self.split = split  # training/testing split
    self.training_size = int(self.num_samples * self.split)
    self.testing_size = self.num_samples - self.training_size

    self.chord_classes = ["major", "minor", "dim", "sus2", "sus4", "dom7", "min7", "maj7"]
    self.chord_vectors = make_categories(self.chord_classes)

    self.root_classes = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
    self.root_vectors = make_categories(self.root_classes)

    self.xs = np.empty((self.num_samples, self.config.rows, self.config.cols))
    self.chord_ys = np.empty((self.num_samples, len(self.chord_classes)))
    self.root_ys = np.empty((self.num_samples, len(self.root_classes)))
    self.freq_ys = np.empty((self.num_samples))
    print("Initializing DeepSamples:DeepChords...")
    print("size", self.config.rows, config.cols)
    print("xs", self.xs.shape)
    print("freq_ys", self.freq_ys.shape)
    print("chord_ys", self.chord_ys.shape)

  def get_config(self):
    return self.config

  def get_chord_classes(self):
    return self.chord_classes

  def get_root_classes(self):
    return self.root_classes

  def load_chords(self):
    print("Loading sample files...")
    files = os.listdir(self.dir_name)
    print("Shuffling samples...")
    np.random.shuffle(files)
    self.files = files
    print("Generating spectrograms...")
    for i, file in enumerate(files[:self.num_samples]):
      self.xs[i] = self.spectrogram(os.path.join(self.dir_name, file))
      self.chord_ys[i] = self.chord_vectors[chord_parts(file)["chord"]]
      self.root_ys[i] = self.root_vectors[chord_parts(file)["root"]]
  
  def load_notes(self):
    print("Loading sample files...")
    files = os.listdir(self.dir_name)
    print("Shuffling samples...")
    np.random.shuffle(files)
    print("Generating spectrograms...")
    for i, file in enumerate(files[:self.num_samples]):
      self.xs[i] = self.spectrogram(os.path.join(self.dir_name, file))
      self.root_ys[i] = self.root_vectors[note_parts(file)["root"]]
      self.freq_ys[i] = note_parts(file)["freq"]

    self.freq_mean = np.mean(self.freq_ys) 
    self.freq_std = np.std(self.freq_ys)
    self.freq_ys -= self.freq_mean
    self.freq_ys /= self.freq_std

  def reshaped_xs(self):
    print("Reshaping for convolutional layers...")
    # self.xs = self.xs / 255.0 # no need to normalize spectrogram output
    # Conv2D layers need an additional dimension for channels. We have just one channel
    # for audio data.
    return self.xs.reshape(self.num_samples, self.config.rows, self.config.cols, 1)

  def get_chord_samples(self):
    (training_xs, training_ys) = (self.xs[:self.training_size], self.chord_ys[:self.training_size])
    (testing_xs, testing_ys) = (self.xs[self.training_size:], self.chord_ys[self.training_size:])
    return ((training_xs, training_ys), (testing_xs, testing_ys))

  def get_root_samples(self):
    (training_xs, training_ys) = (self.xs[:self.training_size], self.root_ys[:self.training_size])
    (testing_xs, testing_ys) = (self.xs[self.training_size:], self.root_ys[self.training_size:])
    return ((training_xs, training_ys), (testing_xs, testing_ys))

  def get_freq_samples(self):
    (training_xs, training_ys) = (self.xs[:self.training_size], self.freq_ys[:self.training_size])
    (testing_xs, testing_ys) = (self.xs[self.training_size:], self.freq_ys[self.training_size:])
    return ((training_xs, training_ys), (testing_xs, testing_ys))

  def spectrogram(self, file):
    return gen_spectrogram(file, self.config)
