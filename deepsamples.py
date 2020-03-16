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

def chord_parts(f):
    parts = f.split("-")
    partmap = {
        "note": parts[0],
        "patch": parts[1],
        "chord": parts[2],
        "inversion": parts[3],
        "freq": float(parts[4]),
        "shift": parts[5],
        "volume": parts[6],
        "reject": parts[7],
        "envelope": parts[8],
    }
    partmap["root"] = re.match('([A-G]+[bs]?)(\d*)', parts[0]).group(1)
    return partmap

def make_categories(classes):
    categories = {}
    for i, ccls in enumerate(classes):
        categories[ccls] = to_categorical(i, num_classes=len(classes))
    return categories


class ChordSamples:
  def __init__(self, dir_name="./samples", num_samples=100, config=DefaultConfig):
    self.config = config
    self.dir_name = dir_name  # path to samples
    self.num_samples = num_samples # number of samples to load

    self.chord_classes = ["major", "minor", "dim", "sus2", "sus4", "dom7", "min7", "maj7"]
    self.chord_vectors = make_categories(self.chord_classes)

    self.root_classes = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
    self.root_vectors = make_categories(self.root_classes)

    print("Initializing DeepSamples:ChordSamples...")
    print("rows/cols:", self.config.rows, config.cols)
    
  def process_file(self, file):
      f, t, Sxx = helpers.spectrogram_from_file(os.path.join(self.dir_name, file), config=self.config, render=False)

      # Transpose the data for the 1D convolutions (time on first axis)
      mags = np.absolute(Sxx)[:self.config.rows,:self.config.cols]
      phases = np.angle(Sxx)[:self.config.rows,:self.config.cols]

      [mags, phases] = helpers.clip_by_magnitude(mags, phases, threshold=self.config.clip_magnitude_quantile, clip_mags=False)
      return [mags, phases]

  def load(self):
    self.mags = np.empty((self.num_samples, self.config.rows, self.config.cols))
    self.phases = np.empty((self.num_samples, self.config.rows, self.config.cols))
    self.chord_ys = np.empty((self.num_samples, len(self.chord_classes)))
    self.root_ys = np.empty((self.num_samples, len(self.root_classes)))
    print("Loading sample files...")
    files = os.listdir(self.dir_name)
    print("Shuffling samples...")
    np.random.shuffle(files)
    self.files = files
    print("Generating spectrograms...")
    for i, file in enumerate(files[:self.num_samples]):
      (self.mags[i], self.phases[i]) = self.process_file(file)
      self.chord_ys[i] = self.chord_vectors[chord_parts(file)["chord"]]
      self.root_ys[i] = self.root_vectors[chord_parts(file)["root"]]
    
    print("Normalizing data...")
    (mag_mean, mag_std) = helpers.normalize(self.mags)
    (phase_mean, phase_std) = helpers.normalize(self.phases)

  def spectrogram(self, file):
    return gen_spectrogram(file, self.config)
