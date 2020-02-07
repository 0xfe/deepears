from keras.utils.np_utils import to_categorical
import os
import sys
import re
import tensorflow as tf
import numpy as np
from scipy import signal
from scipy.io import wavfile


class DeepEars:
    def __init__(self, dir_name="./samples", num_samples=20000, split=0.8):
        self.dir_name = dir_name  # path to samples
        self.spec_dims = (65, 54)  # dims of spectrogram
        self.num_samples = num_samples  # number of samples to load
        self.split = split  # training/testing split
        self.training_size = int(self.num_samples * 0.8)
        self.testing_size = self.num_samples - self.training_size

        self.chord_classes = ["major", "minor", "dim", "dom7", "maj7"]
        self.chord_vectors = {}
        for i, ccls in enumerate(self.chord_classes):
            self.chord_vectors[ccls] = to_categorical(
                i, num_classes=len(self.chord_classes))

        self.root_classes = ["C", "Cs", "D", "Ds",
                             "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
        self.root_vectors = {}
        for i, ccls in enumerate(self.root_classes):
            self.root_vectors[ccls] = to_categorical(
                i, num_classes=len(self.root_classes))

        self.xs = np.empty(
            (self.num_samples, self.spec_dims[0], self.spec_dims[1]))
        self.chord_ys = np.empty((self.num_samples, len(self.chord_classes)))
        self.root_ys = np.empty((self.num_samples, len(self.root_classes)))

    def get_dims(self):
        return self.spec_dims

    def get_chord_classes(self):
        return self.chord_classes

    def get_root_classes(self):
        return self.root_classes

    @staticmethod
    def parts(f):
        parts = f.split("-")
        partmap = {
            "note": parts[0],
            "patch": parts[1],
            "chord": parts[2],
            "envelope": parts[3],
        }
        partmap["root"] = re.match('([A-G]+[bs]?)(\d*)', parts[0]).group(1)
        return partmap

    @staticmethod
    def spectrogram(file):
        fs, data = wavfile.read(file)
        f, t, Sxx = signal.spectrogram(data, fs, window=(
            'hann'), nperseg=64, nfft=128, noverlap=16, mode='magnitude')
        return Sxx

    def load(self):
        files = os.listdir(self.dir_name)
        np.random.shuffle(files)
        for i, file in enumerate(files[:self.num_samples]):
            self.xs[i] = DeepEars.spectrogram(
                os.path.join(self.dir_name, file))
            self.chord_ys[i] = self.chord_vectors[DeepEars.parts(file)[
                "chord"]]
            self.root_ys[i] = self.root_vectors[DeepEars.parts(file)["root"]]

        # self.xs = self.xs / 255.0 # no need to normalize spectrogram output
        # Conv2D layers need an additional dimension for channels. We have just one channel
        # for audio data.
        self.xs = self.xs.reshape(
            self.num_samples, self.spec_dims[0], self.spec_dims[1], 1)

    def train_chord_model(self, epochs=30):
        self.chord_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(100, (3, 3), activation='relu', padding='valid', input_shape=(
                self.get_dims()[0], self.get_dims()[1], 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                200, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.get_chord_classes()), activation='softmax')])

        self.chord_model.compile(
            optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

        (training_xs, training_ys) = (
            self.xs[:self.training_size], self.chord_ys[:self.training_size])
        (testing_xs, testing_ys) = (
            self.xs[self.training_size:], self.chord_ys[self.training_size:])
        self.chord_model.fit(training_xs, training_ys, epochs=epochs,
                             batch_size=100, validation_data=(testing_xs, testing_ys))

    def train_root_model(self, epochs=30):
        self.root_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(100, (3, 3), activation='relu', input_shape=(
                self.get_dims()[0], self.get_dims()[1], 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(100, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.get_root_classes()), activation='softmax')])

        self.root_model.compile(
            optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

        (training_xs, training_ys) = (
            self.xs[:self.training_size], self.root_ys[:self.training_size])
        (testing_xs, testing_ys) = (
            self.xs[self.training_size:], self.root_ys[self.training_size:])
        self.root_model.fit(training_xs, training_ys, epochs=epochs,
                            batch_size=100, validation_data=(testing_xs, testing_ys))

    def predict_chord(self, file):
        Sxx = DeepEars.spectrogram(file).reshape(1, 65, 54, 1)
        chord_class = self.chord_model.predict_classes(Sxx)
        return self.get_chord_classes()[int(chord_class)]

    def predict_root(self, file):
        Sxx = DeepEars.spectrogram(file).reshape(1, 65, 54, 1)
        root_class = self.root_model.predict_classes(Sxx)
        return self.get_root_classes()[int(root_class)]
