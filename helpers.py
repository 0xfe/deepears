import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from IPython import display

'''
Config is a class to keep track of FFT and spectrogram configurations
'''
class Config:
  def __init__(self,
      rows=513,        # Rows: number of FFT bins + 1
      cols = 327,      # Cols: number of time-slices
      s_nperseg = 32,  # Size of segment windowed and processed (in samples)
      s_nfft = 1024,    # FFT size. If > nperseg, then zero-padded on both sides
      s_noverlap = 24, # STFT overlap, the more overlap, the more cols
      resample = 0,    # Resample (hz): if not zero, resample audio before processing
      log_scale = False, # Return log scaled data
      no_spectrogram = False, # Return time-domain data (resampled to rows * cols data points)
      slice_start_s = 0, # Slice start time (seconds)
      slice_duration_s = 0 # Slice duration (seconds)
  ):
    self.rows = rows
    self.cols = cols
    self.s_nperseg = s_nperseg
    self.s_nfft = s_nfft
    self.s_noverlap = s_noverlap
    self.log_scale = log_scale
    self.resample = resample
    self.no_spectrogram = no_spectrogram
    self.slice_start_s = slice_start_s
    self.slice_duration_s = slice_duration_s
    
config = Config(
    rows=129,
    cols = 71,
    s_nperseg = 256,
    s_nfft = 256,
    s_noverlap = 32,
    log_scale = True,
    resample = 16000)

DefaultConfig = config
    
def play_file(file, config=DefaultConfig):
  fs, data = wavfile.read(file)
  if config.resample > 0:
    number_of_samples = round(len(data) * float(config.resample) / fs)
    data = signal.resample(data, number_of_samples)
    fs = config.resample

  display.display(display.Audio(data, rate=fs))

def spectrogram_from_file(file=None, data=None, fs=None, transpose=False, render=True, config=DefaultConfig):
  if data is None:
    fs, data = wavfile.read(file)
    if config.resample > 0:
      number_of_samples = round(len(data) * float(config.resample) / fs)
      data = signal.resample(data, number_of_samples)
      fs = config.resample

  if config.slice_start_s > 0:
    start_sample = config.slice_start_s * fs
    end_sample = start_sample + (config.slice_duration_s * fs)
    data = data[start_sample:end_sample]

  f, t, Sxx = signal.spectrogram(data, fs,
          window=('hann'),
          nperseg=config.s_nperseg,
          nfft=config.s_nfft,
          noverlap=config.s_noverlap,
          mode='complex')
  
  xlabel = 'Frequency (Hz)'
  ylabel = 'Time (sec)'

  if transpose:
    (f, t) = (t, f)
    (xlabel, ylabel) = (ylabel, xlabel)
    Sxx = np.transpose(Sxx)

  if render:
    print("Spectrogram for", file)
    mags = np.absolute(Sxx) 

    if config.log_scale:
      np.log(mags, out=mags)
  
    plt.pcolormesh(t, f, mags, cmap='viridis')
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    print(Sxx.shape)
  
  return f, t, Sxx

def spectrogram_to_audio(Sxx, config=DefaultConfig):
  t, x = signal.istft(Sxx, config.resample,
       window=('hann'),
       nperseg=config.s_nperseg,
       nfft=config.s_nfft,
       noverlap=config.s_noverlap)
  
  display.display(display.Audio(x, rate=config.resample))
  return t, x

def plot_history(history):
  loss = history.history['loss']
  epochs = range(1, len(loss) + 1)

  plt.plot(epochs, loss, 'g-', label='Training loss')
  plt.title('Training loss')
  plt.legend()
  plt.show()

def normalize(x):
  std = np.std(x)
  mean = np.mean(x)
  x -= mean
  x /= std
  return (mean, std)

def denormalize(x, mean, std):
  return (x * std) + mean

def make_windows(xs, length=10, config=config):
  windows = util.view_as_windows(xs, (length, config.rows * 2), 1)
  return np.reshape(windows, (windows.shape[0], length, config.rows * 2))


def clip_by_magnitude(mags, phases, threshold=0.75):
  clip = np.quantile(mags, threshold)
  mags = np.where(np.abs(mags) >= clip, mags, 0)
  phases = np.where(np.abs(mags) >= clip, phases, 0)
  return clip