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
    @classmethod
    def note(cls, str):
        matches = re.match('^([ABCDEFGabcdefg])([b#]?)([0-9])$', str)

        note = matches.group(1).lower()
        acc = matches.group(2).lower()
        octave = int(matches.group(3))

        shift = 0
        if acc == "b":
            shift -= 1
        elif acc == '#':
            shift += 1

        value = ((octave+1) * 12) + cls.values[note] + shift
        return int(value)

class Chord:
    # Intervals
    IU = 0
    Im2 = 1
    IM2 = 2
    Im3 = 3
    IM3 = 4
    Ip4 = 5
    Id5 = 6 # diminished 5th
    Ip5 = 7 # perfect 5th
    Ia5 = 8 # augmented 5th
    Im6 = 8 # minor 6th
    IM6 = 9
    Im7 = 10
    IM7 = 11
    I8 = 12
    Ib9 = I8 + Im2
    I9 = I8 + IM2
    Is9 = I8 + Im3
    I11 = I8 + Ip4
    Is11 = I8 + Id5
    Ib13 = I8 + Im6
    I13 = I8 + IM6

    # Chords
    Maj = [IM2, Ip5]
    Min = [Im3, Ip5]
    Dim = [Im3, Id5]
    Aug = [IM3, Ia5]
    Sus2 = [IM2, Ip5]
    Sus4 = [Ip4, Ip5]
    P5 = [Ip5]
    Maj7 = Maj + [IM7]
    Dom7 = Maj + [Im7]
    Min7 = Min + [Im7]
    MinMaj7 = Min + [IM7]
    Maj9 = Maj7 + [I9]
    Maj7s11 = Maj7 + [Is11]
    Dom7s11 = Dom7 + [Is11]
    Dom7add13 = Dom7 + [I13]
    Maj13 = Maj9 + [I13]
    Min9 = Min7 + [I9]
    Min11 = Min9 + [I11]
    Min13 = Min11 + [I13]
    Dom7s5 = [IM3, Ia5, Im7]
    Dom7b5 = [IM3, Id5, Im7]
    Dom7s5b9 = Dom7s5 + [Ib9]
    Dom7s5s9 = Dom7s5 + [Is9]
    Dom7b5b9 = Dom7b5 + [Ib9]
    Dom7b5s9 = Dom7b5 + [Is9]

    Names = {
        "Maj": {"tones": Maj, "label": "Maj"},
        "Min": {"tones": Min, "label": "Min"},
        "Dim": {"tones": Dim, "label": "Dim"},
        "Aug": {"tones": Aug, "label": "Aug"},
        "Sus2": {"tones": Sus2, "label": "Sus"},
        "Sus4": {"tones": Sus4, "label": "Sus"},
        "P5": {"tones": Maj, "label": "P5"},
        "Maj7": {"tones": Maj7, "label": "Maj"},
        "Dom7": {"tones": Dom7, "label": "Dom"},
        "Min7": {"tones": Min7, "label": "Min"},
        "MinMaj7": {"tones": MinMaj7, "label": "Min"},
        "Maj9": {"tones": Maj9, "label": "Maj"},
        "Maj7s11": {"tones": Maj7s11, "label": "Maj"},
        "Dom7s11": {"tones": Dom7s11, "label": "Dom"},
        "Dom7add13": {"tones": Dom7add13, "label": "Dom"},
        "Maj13": {"tones": Maj13, "label": "Maj"},
        "Min9": {"tones": Min9, "label": "Min"},
        "Min11": {"tones": Min11, "label": "Min"},
        "Min13": {"tones": Min13, "label": "Min"},
        "Dom7s5": {"tones": Dom7s5, "label": "Dom"},
        "Dom7b5": {"tones": Dom7b5, "label": "Dom"},
        "Dom7s5b9": {"tones": Dom7s5b9, "label": "Dom"},
        "Dom7s5s9": {"tones": Dom7s5s9, "label": "Dom"},
        "Dom7b5b9": {"tones": Dom7b5b9, "label": "Dom"},
        "Dom7b5s9": {"tones": Dom7b5s9, "label": "Dom"}
    }

    @classmethod
    def gen_chord(cls, intervals, root):
        return [root] + map(lambda i: root + i, intervals)
    
class Track:
    def __init__(self, name):
        self.name = name
        self.events = []
        self.time = 0     # in beats
        self.tempo = 0
        self.program = 0

    def set_tempo(self, tempo):
        self.tempo = tempo

    def set_program(self, program):
        self.program = program

    def add_note(self, pitch, duration, volume=100):
        self.events.append({
            "type": "note",
            "channel": 0,
            "pitch": pitch,
            "time": self.time,
            "duration": duration,
            "volume": volume
        })
        self.time += duration

    def add_chord(self, pitches, duration, volume=100):
        channel = 0
        for pitch in pitches:
            self.events.append({
                "type": "note",
                "channel": channel,
                "pitch": pitch,
                "time": self.time,
                "duration": duration,
                "volume": volume
            })
            self.time += duration
            channel += 1

    def write_to(self, song, track_num):
        song.addTempo(track_num, self.time, self.tempo)
        for event in self.events:
            song.addNote(track_num, event["channel"], event["pitch"],
                         event["time"], event["duration"], event["volume"])

class Song:
    DEBUG = True

    def __init__(self):
        self.tracks = []

    def new_track(self, name):
        track = Track(name)
        self.tracks.append(track)
        return track

    def write(self, file_name):
        file = MIDIFile(len(self.tracks))
        for x in range(0, len(self.tracks)):
            self.tracks[x].write_to(file, x)

        with open(file_name, "wb") as out:
            file.writeFile(out)
    
song = Song()
track = song.new_track("Test")
track.set_tempo(60)
track.set_program(0)

# Add a chord
track.add_chord(Chord.gen_chord(Chord.Min, Note.note("C4")), 1)

# Add some notes
N = Note.note
degrees = [N("C4"), N("D4"), N("E4")]
for pitch in degrees:
    track.add_note(pitch, 1)

song.write("minor-scale.mid")
