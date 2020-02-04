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
