# DeepEars

Mohit Muthanna Cheppudira <mohit@muthanna.com>

Various experiments in music analysis using deep learning.

Ignore the `src/` directory -- old junk.

## Generate Training Data

```
./gen_{chord|note}_samples
zip -r samples.zip samples/
gsutil -h "Cache-control:public,max-age=86400" -m cp -a public-read samples.zip gs://muthanna.com/deepears/instrument-samples-v1.zip
```

Some notebooks let you generate training data from within the book.

## Build and train

Use Jupyter to open one of the ipynb notebooks and follow instructions there.

Current notebooks:

* DeepChords - Chord and root recognition model
* Polyphonic Pitch (Multi GPU) - Detect multiple pitches for the same instrument
* Phase_Reconstruction - Attempt to reconstruct phases for magnitude-only spectrograms
* DeepInstruments GAN (Multi GPU) - Generate new instruments, uses distributed training
* DeepSTFT - Teach neural networks to learn STFT and FFT functions
* DeepEars - Old models for chord, pitch, and root detection

