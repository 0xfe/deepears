#!/usr/bin/env python3

import os
import re
import tempfile
import math
import random
import numpy as np
from midiutil.MidiFile import MIDIFile
from midi import Note, GM_PATCHES


class Sample:
    SOUNDFONT = "soundfont.sf2"

    @staticmethod
    def note_to_freq(note, octave, a_440=440.0):
        note_map = {}
        for i, note_name in enumerate(Note.names):
            note_map[note_name] = i

        key = note_map[note] + ((octave - 1) * 12) + 1
        return a_440 * pow(2, (key - 46) / 12.0)

    def __init__(self, name, program=0, key='A', octave=4, volume=100, tempo=60):
        self.name = name
        self.mid_filename = name + '.mid'
        self.tmp_filename = self.name + "-" + "full" + ".wav"

        self.key = key
        self.octave = octave
        self.note = str(key)+str(octave)
        self.freq = Sample.note_to_freq(key, octave)

        self.file = MIDIFile(1, adjust_origin=True)
        self.file.addTempo(0, 0, tempo)
        self.file.addProgramChange(0, 0, 0, program)
        self.file.addNote(0, 0, Note.note(self.note), 0, 1, volume)

    def save(self):
        with open(self.mid_filename, "wb") as out:
            self.file.writeFile(out)

    def make_wav(self):
        self.save()
        os.system("fluidsynth -l -i -a file %s %s -F %s -r 44100" %
                  (Sample.SOUNDFONT, self.mid_filename, self.tmp_filename))

    def transform_wav(self, suffix, start_s=0, duration=1,
                      pitch_shift_hz=0, resample_hz=44100, resample_bits=16,
                      volume=1.0, strip_fundamental=False):
        shift_cents = 0
        if pitch_shift_hz != 0:
            shift_cents = 1200.0 * \
                math.log((self.freq + pitch_shift_hz) / self.freq, 2)

        self.new_freq = self.freq + pitch_shift_hz

        bandreject = ""
        bandreject_ext = "noreject"
        if strip_fundamental:
            bandreject = "bandreject %f 0.1o" % self.new_freq
            bandreject_ext = "reject"

        wav_file = self.name + "-" + \
            "%.3f" % (self.new_freq) + "-" + \
            "S" + str(pitch_shift_hz) + "-" + \
            "V" + str(volume) + "-" + \
            str(bandreject_ext) + "-" + \
            suffix + ".wav"

        print("Writing ", wav_file, "with sample rate",
              resample_hz, "and bit depth", resample_bits, ("(shift %s)" % shift_cents))

        os.system("sox -t raw -r 44100 -e signed -b 16 -c 2 -v %f %s -r %s -b %s %s norm -0.1 pitch %s remix 2 %s trim %f %f" %
                  (volume, self.tmp_filename, resample_hz, resample_bits, wav_file, shift_cents, bandreject, start_s, duration))

    def clean(self):
        # Keep only transformed
        os.remove(self.mid_filename)  # Remove midi file
        os.remove(self.tmp_filename)  # Remove -full .wav file


# Generate samples of single note files for pitch detection.
def gen_note_samples():
    for octave in range(2, 8):
        random.shuffle(GM_PATCHES)
        for patch in range(0, 15):
            program = GM_PATCHES[patch]
            for key in Note.names:
                sample = Sample("samples/note-%s%s-P%s" % (key, octave, program),
                                program=program, key=key, octave=octave)
                sample.make_wav()
                for pitch_shift_hz in np.concatenate((np.array([0]), np.random.randint(10, 80, 5))):
                    volume = np.round(np.random.uniform(0.2, 1.0), decimals=2)
                    strip_fundamental = np.random.uniform() > 0.8
                    sample.transform_wav(
                        "attack", duration=0.33, pitch_shift_hz=pitch_shift_hz, volume=volume, strip_fundamental=strip_fundamental)
                    sample.transform_wav(
                        "sustain", start_s=0.33, duration=0.33, pitch_shift_hz=pitch_shift_hz, volume=volume, strip_fundamental=strip_fundamental)
                    sample.transform_wav(
                        "decay", start_s=0.66, duration=0.33, pitch_shift_hz=pitch_shift_hz, volume=volume, strip_fundamental=strip_fundamental)

                sample.clean()


# Generate samples of instrument voices for GAN
def gen_instrument_samples():
    for octave in range(2, 7):
        random.shuffle(GM_PATCHES)
        for program in GM_PATCHES:
            for key in Note.names:
                sample = Sample("samples/note-%s%s-P%s" % (key, octave, program),
                                program=program, key=key, octave=octave)
                sample.make_wav()
                for pitch_shift_hz in np.concatenate((np.array([0]), np.random.randint(10, 80, 5))):
                    volume = np.round(np.random.uniform(0.2, 1.0), decimals=2)
                    sample.transform_wav("instrument", duration=1,
                                         pitch_shift_hz=pitch_shift_hz, volume=volume,
                                         strip_fundamental=False)
                sample.clean()


if __name__ == "__main__":
    gen_instrument_samples()
