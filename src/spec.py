from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

if len(sys.argv) < 2:
    print("Usage: spec.py filename")
    sys.exit(1)

filename = sys.argv[1]

# Returns sample rate, data bytes
# A 1 second 10khz wave file at unsigned 8-bit sample rate will return 10000 8-bit bytes.
fs, data = wavfile.read('audio.wav')
print(data.shape)

f, t, Sxx = signal.spectrogram(data, fs, nfft=1024)
print(Sxx.shape)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.show()
