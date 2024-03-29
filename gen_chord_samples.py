#!/usr/bin/env python3

import numpy as np
from midi import Sample, Chord, Note, GM_PATCHES
import random
from IPython import display
from IPython.utils import io
import multiprocessing as mp

OUTDIR = "./samples"
OCTAVES = range(2, 6)
NUM_PATCHES = 20
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


def write_chord_samples(root, octave, chord, inversion, program, resample_hz=44100, resample_bits=16):
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
                        write_chord_samples(root, octave, chord,
                                            inversion, program, resample_hz=16000)


def write_polyphonic_samples(notes, program, resample_hz=16000, resample_bits=16, num_pitch_shifts=2):
    sample = Sample(
        OUTDIR+"/poly-" +
        "-P" + str(program) +
        "-N:" + (':').join(notes))

    sample.new_track(program)
    sample.write_chord(Note.notes(notes))
    print("Saving:", sample.mid_filename, sample.tmp_filename)
    freq = 440
    sample.make_wav()
    for pitch_shift_hz in np.concatenate((np.array([0]), np.random.randint(-30, 30, num_pitch_shifts))):
        volume = np.round(
            np.random.uniform(0.2, 1.0), decimals=2)
        sample.transform_wav("full", start_s=0, duration=1, pitch_shift_base_freq=freq, volume=volume,
                             pitch_shift_hz=pitch_shift_hz, resample_hz=resample_hz, resample_bits=resample_bits)
    sample.clean()


def gen_polyphonic_samples(num_samples=1, hz=16000, patches_per_sample=5):
    # Generates num_samples * patches_per_sample * (num_pitch_shifts + 1) subsamples per sample
    print("Generating %d samples with %d processes..." % (num_samples * patches_per_sample, mp.cpu_count()))
    all_notes = []
    for key in Note.names:
        for octave in range(2, 7):
            all_notes.append("%s%d" % (key, octave))

    random.shuffle(all_notes)
    p = display.ProgressBar(num_samples)
    p.display()
    pool = mp.Pool(mp.cpu_count())
    for i in range(num_samples):
        p.progress = i
        num_notes = np.random.randint(7) + 1  # At most 7 notes (1 - 7)
        notes = [all_notes[np.random.randint(
            len(all_notes))] for j in range(num_notes)]

        # Generate file from note_array
        random.shuffle(GM_PATCHES)
        with io.capture_output() as captured:
            for program in GM_PATCHES[:patches_per_sample]:
                pool.apply(write_polyphonic_samples, args=(notes, program, hz))
                
    pool.close()
    p.progress = num_samples


if __name__ == "__main__":
    gen_polyphonic_samples(1000)
