'''
Python implementation of gen_midi_samples.rb.

Requires:
$ pip install MIDIUtil

'''

import re

from midiutil.MidiFile import MIDIFile

class Note:
    values = {
        "c": 0,
        "d": 2,
        "e": 4,
        "f": 5,
        "g": 7,
        "a": 9,
        "b": 11
    }

    names = ["C", "Cs", "D", "Eb", "E", "F", "Fs", "G", "Ab", "A", "Bb", "B"]

    # Takes a note in the form "C#4", "Bb2", "A5", etc. and returns
    # the MIDI note number.
    def note(str):
        matches = re.match('^([ABCDEFGabcdefg])([b#]?)([0-9])$')

        note = matches.group(1).lower()
        acc = matches.group(2).lower()
        octave = int(matches.group(3))

        shift = 0
        if acc == "b":
            shift -= 1
        else if acc == '#':
            shift += 1

        value = ((octave+1) * 12) + Values[note] + shift
        return int(value.to_i)

degrees  = [60, 62, 64, 65, 67, 69, 71, 72] # MIDI note number
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 60  # In BPM
volume   = 100 # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1) # One track, defaults to format 1 (tempo track
                     # automatically created)
MyMIDI.addTempo(track,time, tempo)

for pitch in degrees:
    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    time = time + 1

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)

