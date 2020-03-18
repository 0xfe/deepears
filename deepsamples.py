import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os
from keras.utils.np_utils import to_categorical
from IPython import display

import helpers
from helpers import Config, DefaultConfig

from midi import Note

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

    self.chord_classes = ["major", "minor", "dim", "dom7", "min7", "maj7"]
    self.chord_vectors = make_categories(self.chord_classes)

    self.root_classes = Note.names
    self.root_vectors = make_categories(self.root_classes)

    print("Initializing DeepSamples:ChordSamples...")
    print("rows/cols:", self.config.rows, config.cols)
    
  def process_file(self, file):
      f, t, Sxx = helpers.spectrogram_from_file(os.path.join(self.dir_name, file), config=self.config, render=False)

      mags = np.absolute(Sxx)[:self.config.rows,:self.config.cols]
      phases = np.angle(Sxx)[:self.config.rows,:self.config.cols]

      [mags, phases] = helpers.clip_by_magnitude(mags, phases, threshold=self.config.clip_magnitude_quantile, clip_mags=False)
      if self.config.log_scale:
        mags = np.log(mags)
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
    p = display.ProgressBar(self.num_samples)
    p.display()
    for i, file in enumerate(files[:self.num_samples]):
      (self.mags[i], self.phases[i]) = self.process_file(file)
      self.chord_ys[i] = self.chord_vectors[chord_parts(file)["chord"]]
      self.root_ys[i] = self.root_vectors[chord_parts(file)["root"]]
      if i % 50 == 0: p.progress = i
  
    p.progress = self.num_samples
    print("Normalizing data...")
    (self.mag_mean, self.mag_std) = helpers.normalize(self.mags)
    (self.phase_mean, self.phase_std) = helpers.normalize(self.phases)
    print("Samples ready.")
    
class PolySamples:
  @classmethod
  def file_parts(cls, f):
    parts = f.split("-")
    notes = parts[3].split(':')[1:]
    partmap = {
        "patch": parts[2],
        "notes": notes,
        "freq": float(parts[4]),
        "shift": parts[5],
        "volume": parts[6],
        "reject": parts[7],
    }
    return partmap

  def __init__(self, dir_name="./samples", num_files=100, config=DefaultConfig):
    self.config = config
    self.dir_name = dir_name  # path to samples
    self.num_files = num_files # number of samples to load

    # Generate 12 tones across 5 octaves
    self.note_classes = ["%s%d" % (key, octave) for octave in range(2,7) for key in Note.names]
    self.note_vectors = make_categories(self.note_classes)
    
    # Slice samples into windows
    self.window_size = 10
    self.window_stride = 5
    self.windows_per_file = ((config.cols - self.window_size) // self.window_stride) + 1
    
    self.rows = self.config.rows
    self.cols = self.window_size

    print("Initializing DeepSamples:PolySamples...")
    print("rows: %d, cols: %d, windows_per_file: %d" % (self.config.rows, config.cols, self.windows_per_file))
  
  def notes_to_vector(self, notes):
    # Returns a multi-label vector for notes, by summing the one-hot encodings
    # and clipping.
    return np.minimum(
      np.sum(np.array([self.note_vectors[note] for note in notes]), axis=0),
      np.ones(len(self.note_classes)))
  
  def vector_to_notes(self, vector):
    indexes = np.where(vector == 1)[0]
    return [self.note_classes[x] for x in indexes]
    
  def process_file(self, file):
    f, t, Sxx = helpers.spectrogram_from_file(os.path.join(self.dir_name, file), config=self.config, render=False)

    mags = np.absolute(Sxx)[:self.config.rows,:self.config.cols]
    phases = np.angle(Sxx)[:self.config.rows,:self.config.cols]

    [mags, phases] = helpers.clip_by_magnitude(mags, phases, threshold=self.config.clip_magnitude_quantile, clip_mags=False)
    
    if self.config.log_scale:
      mags = np.log(mags)
    
    length = mags.shape[1]
    mag_windows = [mags[:,i:i + self.window_size] for i in range(0, length - self.window_size, self.window_stride)]
    phase_windows = [phases[:,i:i + self.window_size] for i in range(0, length - self.window_size, self.window_stride)]

    return [mag_windows, phase_windows]

  def load(self):
    self.tunings = np.empty((self.num_files * self.windows_per_file, 1))
    self.mags = np.empty((self.num_files * self.windows_per_file, self.rows, self.cols))
    self.phases = np.empty((self.num_files * self.windows_per_file, self.rows, self.cols))
    self.ys = np.empty((self.num_files * self.windows_per_file, len(self.note_classes)))

    print("Loading sample files...")
    files = os.listdir(self.dir_name)
    print("Shuffling samples...")
    np.random.shuffle(files)
    self.files = files
    
    print("Generating spectrograms...")
    p = display.ProgressBar(self.num_files)
    p.display()
    for i, file in enumerate(files[:self.num_files]):
      (mag_windows, phase_windows) = self.process_file(file)
      for j in range(len(mag_windows)):
        self.mags[i * self.windows_per_file + j] = mag_windows[j]
        self.phases[i * self.windows_per_file + j] = phase_windows[j]
        
      start = i * self.windows_per_file
      end = start + self.windows_per_file
      
      self.tunings[start:end] = self.file_parts(file)["freq"]
      self.ys[start:end] = self.notes_to_vector(self.file_parts(file)["notes"])

      if i % 50 == 0: p.progress = i
  
    p.progress = self.num_files
    print("Normalizing data...")
    (self.mag_mean, self.mag_std) = helpers.normalize(self.mags)
    (self.phase_mean, self.phase_std) = helpers.normalize(self.phases)
    print("Samples ready.")