#!/usr/bin/env python3

from midi import Sample, Chord, Note, GM_PATCHES
import random

Sample.ENCODE_BITS = 8
Sample.ENCODE_HZ = 8000

OUTDIR = "./samples"

OCTAVES = range(2, 6)
NUM_PATCHES = 10
INVERSIONS = [0, 1, 2]
ROOTS = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
CHORDS = [
    Chord.Names["Maj"],
    Chord.Names["Min"],
    Chord.Names["Dim"],
    Chord.Names["Sus2"],
    Chord.Names["Sus4"],
    Chord.Names["Dom7"],
    Chord.Names["Min7"],
    Chord.Names["Maj7"],
]


def render_chord(chord, octave, name):
    for root in ROOTS:
        for inversion in INVERSIONS:
            base = root + str(octave)
            random.shuffle(GM_PATCHES)
            program = GM_PATCHES[0]

            sample = Sample(
                OUTDIR+"/" +
                base +
                "-P" + str(program) +
                "-" + name +
                "-i" + str(inversion))

            sample.new_track(program)
            sample.write_chord(
                Chord.invert(Chord.chord(chord, Note.note(base)), inversion))
            print("Saving:", sample.file)
            sample.make_wav()
            sample.save_wav("attack", 0, 0.33)
            sample.save_wav("sustain", 0.33, 0.33)
            sample.save_wav("decay", 0.66, 0.33)
            sample.clean()


NUM_SAMPLES = len(OCTAVES) * NUM_PATCHES * \
    len(INVERSIONS) * len(ROOTS) * len(CHORDS)
NUM_SAMPLES = NUM_SAMPLES * 3  # attack, sustain, decay

print("Rendering", NUM_SAMPLES, "samples.")
input("Press Enter to start...")

for octave in OCTAVES:
    for patch in range(0, NUM_PATCHES):  # 10 random patches from GM_PATCHES
        for chord in CHORDS:
            render_chord(chord["tones"], octave, chord["label"])
