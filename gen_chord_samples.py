#!/usr/bin/env python3

import numpy as np
from midi import Sample, Chord, Note, GM_PATCHES
import random

OUTDIR = "./samples"
OCTAVES = range(2, 6)
NUM_PATCHES = 10
INVERSIONS = [0, 1, 2]
ROOTS = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
NUM_ROOTS = 5
CHORDS = [
    Chord.Names["Maj"],
    Chord.Names["Min"],
    Chord.Names["Dim"],
    # Chord.Names["Sus2"],
    # Chord.Names["Sus4"],
    Chord.Names["Dom7"],
    Chord.Names["Min7"],
    Chord.Names["Maj7"],
]


def write_samples(root, octave, chord, inversion, program, resample_hz=44100, resample_bits=16):
    chord_name = chord["label"]
    base = root + str(octave)
    chord_intervals = Chord.chord(
        chord["tones"], Note.note(base))

    sample = Sample(
        OUTDIR+"/" +
        base +
        "-P" + str(program) +
        "-" + chord_name +
        "-i" + str(inversion))

    sample.new_track(program)
    sample.write_chord(Chord.invert(
        chord_intervals, inversion))
    print("Saving:", sample.mid_filename,
          sample.tmp_filename)
    freq = Sample.note_to_freq(root, octave)
    sample.make_wav()
    # np.concatenate((np.array([0]), np.random.randint(10, 80, 5))):
    for pitch_shift_hz in [0]:
        volume = np.round(
            np.random.uniform(0.2, 1.0), decimals=2)
        sample.transform_wav(
            "attack", start_s=0, duration=0.33, pitch_shift_base_freq=freq, volume=volume, pitch_shift_hz=pitch_shift_hz, resample_hz=resample_hz, resample_bits=resample_bits)
        sample.transform_wav(
            "sustain", start_s=0.33, duration=0.33, pitch_shift_base_freq=freq, volume=volume, pitch_shift_hz=pitch_shift_hz, resample_hz=resample_hz, resample_bits=resample_bits)
        sample.transform_wav(
            "decay", start_s=0.66, duration=0.33, pitch_shift_base_freq=freq, volume=volume, pitch_shift_hz=pitch_shift_hz, resample_hz=resample_hz, resample_bits=resample_bits)
    sample.clean()


def gen_chord_samples():
    NUM_SAMPLES = len(OCTAVES) * NUM_PATCHES * \
        len(INVERSIONS) * NUM_ROOTS * len(CHORDS)
    NUM_SAMPLES = NUM_SAMPLES * 3 * 1  # evelope, pitch_shift

    print("Rendering", NUM_SAMPLES, "samples.")
    input("Press Enter to start...")

    for octave in OCTAVES:
        for chord in CHORDS:
            random.shuffle(ROOTS)
            for root in ROOTS[:NUM_ROOTS]:
                random.shuffle(GM_PATCHES)
                for program in GM_PATCHES[:NUM_PATCHES]:
                    for inversion in INVERSIONS:
                        write_samples(root, octave, chord,
                                      inversion, program, resample_hz=16000)


if __name__ == "__main__":
    gen_chord_samples()
