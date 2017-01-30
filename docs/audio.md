## MIDI Sample Generation

### Installation (Needs Ruby 2.0)

    $ gem install midilib
    $ brew install fluidsynth sox jq
    Linux: $ apt-get install fluidsynth sox jq

    For python version (incomplete): $ pip install MIDIlib

### Get soundfont (see AWS setup instructions below)

    $ pip install awscli
    $ gpg2 -d awskey.csv.gpg
    $ aws configure (use output from above, region: us-east-1)
    $ aws s3 cp s3://tftrain/soundfont.sf2 .

### Generate MIDI

    $ ./src/gen_midi_samples.rb
    $ play data/wav/*`

### To play

    $ fluidsynth -a coreaudio FluidR3_GM.sf2 "/Applications/Band-in-a-Box/Styles/Ear Training/Music Replay/MelodyReplay/MR02041.MID"

### To convert to WAV

    $ fluidsynth -l -i -a file ~/w/audio/octave/FluidR3_GM.sf2 from_scratch.mid -F raw_audio
    $ sox -t raw -r 44100 -e signed -b 16 -c 2 raw_audio audio.wav

### Mix down to 1 channel and normalize

    $ sox -t raw -r 44100 -e signed -b 16 -c 2 test.raw ~/Downloads/test1.wav norm remix 2

(If you get clipping, try -b 32. If that doesn’t work add —norm)

### To create spectrogram

    $ sox ~/Downloads/test1.wav -n remix 2 spectrogram -x 400 -y 400 -m -r

(-m: monochrome, -r: raw, don’t show axis, remix: mix down two channels to 1)

Only first three seconds (trim):

    $ sox ~/Downloads/test1.wav -n remix 2 trim 0 3 spectrogram -x 400 -y 400 -m -o spectrogram.png

Quantize to 128 levels:

    $ sox ~/Downloads/test1.wav -n remix 2 trim 0 1 spectrogram -x 200 -y 200 -m -r -q 128 -o spectrogram.png

To add noise:

    $ sox ~/Downloads/test1.wav -p synth whitenoise vol 0.1 | sox -m ~/Downloads/test1.wav - ~/Downloads/noisy.wav
