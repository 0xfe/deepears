#!/usr/bin/env python
#
# Calculates FFT of a section of a WAV file. Does not
# work on 24-bit files.
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
# To run:
#
# First read README.md and install virtualenv
#
#    $ source ./tf/bin/activate
#    $ ./fft
import scipy, numpy as np
from scipy.io import wavfile
from scipy import stats

# This import ordering is important for virtualenv. Otherwise
# it looks for an interactive display which isn't available.
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

FILE = './data/wav/singles-V37-P30-O2-N1-B9000-attack.wav'

# Open wav audio 'file' and do an N-point FFT of
# data starting at START. Returns the sample rate and
# an array of  N/2 imaginary numbers representing the freq domain.
def fft_wav(file, N, START):
    rate, data = wavfile.read(file)
    # Calculate FFT for real valued input (rfft). Should return
    # N/2 imaginary numbers.
    return rate, np.fft.rfft(data[START:START+(N-1)])

def plot_fft(fft, rate, file):
    N = fft.size * 2

    # Create the x-axis for N/2 bins and scale it to associate each
    # bin with its center-frequency.
    xaxis = np.array([x for x in range(0, N/2)]) * (rate / N)

    # Amplitude per bin is the magnitude of the complex value of
    # the bin, divided by the number of bins (power distributed across bins.) 
    amplitudes = (2.0 / N) * np.abs(y[0:N/2])

    # Plot frequency histogram.
    plt.plot(xaxis, amplitudes)
    plt.savefig(file)


# Returns a 2D array of STFT frames. Each frame represents a time duration
# of N samples, and consists of N/2 complex numbers, one per frequency bin.
#
# The overlap represents the amount of time overlap between frames. So a value
# of 4 means 25% overlapping frames.
def stft(x, fftsize=2048, overlap=4):   
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def plot_spectrogram(audio_data, rate, fftsize=256, overlap=4, file="stft.png"):
    N = fftsize
    sy = stft(audio_data, N, overlap)   # sy.shape = [N/2, duration/hops]

    # Calculate spectrogram based on magnitudes of complex components.
    sgram = scipy.absolute(sy.T) * (2.0/N)

    # DEBUGGING:
    # length_x = sgram.shape[1]                       # number of x axis bins (duration)
    # length_s = (audio_data.size / float(rate))      # duration in seconds
    # x_bins = np.linspace(0, length_s, length_x)     # spread duration into bins
    # y_bins = np.linspace(0, rate/2, (N/2 + 1))      # spread frequency into bins
    # print(x_bins, y_bins) # Show bin values

    # plt.imshow(sgram, origin="lower", cmap=plt.get_cmap("Blues"))
    plt.figure(1)
    plt.subplot(211)
    plt.ylabel("log2")
    plt.imshow(np.log2(sgram), origin="lower", cmap=plt.get_cmap("Blues"))
    plt.subplot(212)
    plt.ylabel("linear")
    plt.imshow(sgram, origin="lower", cmap=plt.get_cmap("Blues"))
    plt.savefig(file)

# rate, y = fft_wav(FILE, 2048, 5000)
# plot_fft(y, rate, "fft.png")

rate, audio_data = wavfile.read(FILE)
print stats.describe(np.array(audio_data))
print np.std(np.array(audio_data))
plot_spectrogram(audio_data, rate, 512)
