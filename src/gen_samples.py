#!/usr/bin/env python3

from midi import Sample, Chord, Note, GM_PATCHES

sample = Sample("c-major.mid")
sample.write_chord(Chord.chord(Chord.Maj, Note.note("C4")))
sample.new_track(19)  # Church organ
sample.write_chord(Chord.chord(Chord.Maj, Note.note("C5")))
sample.save()

sample = Sample("c-scale.mid")
sample.write_notes(Note.notes(["C2", "C3", "C4", "C5", "C6", "C7", "C8"]))
sample.save()

OUTDIR = "./samples"


def render_chord(chord, octave, program, name):
    for root in ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]:
        base = root + str(octave)
        sample = Sample(OUTDIR+"/"+base+"-"+name)
        sample.new_track(19)
        sample.write_chord(Chord.chord(chord, Note.note(base)))
        print("Saving:", sample.file)
        sample.make_wav()
        sample.save_wav("attack", 0, 0.33)
        sample.save_wav("sustain", 0.33, 0.33)
        sample.save_wav("decay", 0.66, 0.33)


for octave in range(2, 8):
    for patch in GM_PATCHES:
        render_chord(Chord.Maj, octave, patch, "P"+str(patch)+"-major")
        render_chord(Chord.Min, octave, patch, "P"+str(patch)+"-minor")
