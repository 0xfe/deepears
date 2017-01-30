#!/usr/bin/env python

from midi import Sample, Chord

sample = Sample("c-major.mid")
sample.write_chord(Chord.Maj, "C4")