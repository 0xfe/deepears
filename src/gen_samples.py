#!/usr/bin/env python

from midi import Sample, Chord, Note

sample = Sample("c-major.mid")
sample.write_chord(Chord.chord(Chord.Maj, Note.note("C4")))
sample.new_track(19)  # Church organ
sample.write_chord(Chord.chord(Chord.Maj, Note.note("C5")))
sample.save()

sample = Sample("c-scale.mid")
sample.write_notes(Note.notes(["C2", "C3", "C4", "C5", "C6", "C7", "C8"]))
sample.save()

OUTDIR = "./samples"


def render_chord(intervals, octave, name):
    for root in ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]:
        base = root + str(octave)
        sample = Sample(OUTDIR+"/"+base+"-"+name)
        sample.write_chord(Chord.chord(intervals, Note.note(base)))
        print("Saving:", sample.file)
        sample.make_wav()
        sample.save_wav("attack", 0, 0.33)
        sample.save_wav("sustain", 0.33, 0.33)
        sample.save_wav("decay", 0.66, 0.33)


for octave in range(4, 5):
    render_chord(Chord.Maj, octave, "major")
    render_chord(Chord.Min, octave, "minor")
