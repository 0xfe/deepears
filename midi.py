'''
Python MIDI abstractions for generating synthetic audio
samples. Only implements the minimum required for simple
audio snippets.

Requires:
$ pip install MIDIUtil
'''

import os
import re
import math
import tempfile
from midiutil.MidiFile import MIDIFile

GM_PATCHES = [
    0,  # Acoustic Grand Piano
    3,  # Honky-tonk Piano
    6,  # Harpsichord
    11,  # Vibraphone
    16,  # Drawbar Organ
    19,  # Church Organ
    22,  # Harmonica
    24,  # Acoustic Guitar (nylon)
    26,  # Electric Guitar (jazz)
    30,  # Distortion Guitar
    32,  # Acoustic Bass
    33,  # Electric Bass (finger)
    40,  # Violin,
    42,  # Cello
    48,  # String Ensemble 1
    51,  # SynthStrings 2
    52,  # Choir Aahs
    56,  # Trumpet
    57,  # Trombone
    61,  # Brass Section
    65,  # Alto Sax
    66,  # Tenor Sax
    71,  # Clarinet
    73,  # Flute
    78  # Whistle
]


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

    names = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

    # Takes a note in the form "C#4", "Bb2", "A5", etc. and returns
    # the MIDI note number.
    @classmethod
    def note(cls, str):
        matches = re.match('^([ABCDEFGabcdefg])([b#s]?)([0-9])$', str)

        note = matches.group(1).lower()
        acc = matches.group(2).lower()
        octave = int(matches.group(3))

        shift = 0
        if acc == "b":
            shift -= 1
        elif acc == '#':
            shift += 1
        elif acc == 's':
            shift += 1

        value = ((octave+1) * 12) + cls.values[note] + shift
        return int(value)

    @classmethod
    def notes(cls, note_list):
        return map(cls.note, note_list)


class Chord:
    # Intervals
    IU = 0
    Im2 = 1
    IM2 = 2
    Im3 = 3
    IM3 = 4
    Ip4 = 5
    Id5 = 6  # diminished 5th
    Ip5 = 7  # perfect 5th
    Ia5 = 8  # augmented 5th
    Im6 = 8  # minor 6th
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
    Maj = [IM3, Ip5]
    Min = [Im3, Ip5]
    Dim = [Im3, Id5]
    Sus2 = [IM2, Ip5]
    Sus4 = [Ip4, Ip5]
    Dom7 = Maj + [Im7]
    Min7 = Min + [Im7]
    Maj7 = Maj + [IM7]
    Aug = [IM3, Ia5]
    P5 = [Ip5]
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
        "Maj": {"tones": Maj, "label": "major"},
        "Min": {"tones": Min, "label": "minor"},
        "Dim": {"tones": Dim, "label": "dim"},
        "Sus2": {"tones": Sus2, "label": "sus2"},
        "Sus4": {"tones": Sus4, "label": "sus4"},
        "Dom7": {"tones": Dom7, "label": "dom7"},
        "Min7": {"tones": Min7, "label": "min7"},
        "Maj7": {"tones": Maj7, "label": "maj7"},

        # Unused (for now)
        "MinMaj7": {"tones": MinMaj7, "label": "minmaj7"},
        "Maj9": {"tones": Maj9, "label": "maj9"},
        "Aug": {"tones": Aug, "label": "aug"},
        "P5": {"tones": P5, "label": "p5"},

        # Set labels below
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
    def chord(cls, intervals, root):
        return [root] + list(map(lambda i: root + i, intervals))

    @classmethod
    def invert(cls, notes, inversion):
        for i in range(0, inversion):
            notes[i] += 12
        return notes


class Track:
    def __init__(self, name):
        self.name = name
        self.events = []
        self.time = 0     # in beats
        self.tempo = 0
        self.program = 0
        self.num_channels = 1
        self.tunings = []

    def set_tempo(self, tempo):
        self.tempo = tempo

    def set_program(self, program):
        self.program = program

    def set_tunings(self, tunings):
        self.tunings = tunings

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
            channel += 1
        self.num_channels = channel

    def write_to(self, backend, track_num):
        backend.addTempo(track_num, self.time, self.tempo)
        for channel in range(0, self.num_channels):
            backend.addProgramChange(track_num, channel, 0, self.program)
            if len(self.tunings) > 0:
                backend.changeNoteTuning(
                    track_num, self.tunings, tuningProgam=self.program)
                backend.changeTuningBank(track_num, channel, 0, 0)
                backend.changeTuningProgram(
                    track_num, channel, 0, self.program)

        for event in self.events:
            backend.addNote(track_num, event["channel"], event["pitch"],
                            event["time"], event["duration"], event["volume"])


class Song:
    def __init__(self):
        self.tracks = []

    def new_track(self, name):
        track = Track(name)
        self.tracks.append(track)
        return track

    def write(self, file_name):
        file = MIDIFile(len(self.tracks), adjust_origin=True)
        for x in range(0, len(self.tracks)):
            self.tracks[x].write_to(file, x)

        with open(file_name, "wb") as out:
            file.writeFile(out)


class Sample:
    BEATS = 1
    TEMPO = 60
    SOUNDFONT = "soundfont.sf2"
    ENCODE_BITS = 16
    ENCODE_HZ = 44100

    @staticmethod
    def note_to_freq(note, octave, a_440=440.0):
        note_map = {}
        for i, note_name in enumerate(Note.names):
            note_map[note_name] = i

        key = note_map[note] + ((octave - 1) * 12) + 1
        return a_440 * pow(2, (key - 46) / 12.0)

    def __init__(self, name):
        self.song = Song()
        self.name = name
        self.mid_filename = name + '.mid'
        self.tmp_filename = self.name + "-" + "full" + ".wav"
        self.new_track()

    def new_track(self, program=0):
        self.track = self.song.new_track("Sample")
        self.track.set_tempo(Sample.TEMPO)
        self.track.set_program(program)
        return self.track

    def write_chord(self, notes):
        self.track.add_chord(notes, Sample.BEATS)

    def write_notes(self, notes):
        for pitch in notes:
            self.track.add_note(pitch, Sample.BEATS)

    def save(self):
        self.song.write(self.mid_filename)

    def make_wav(self):
        self.save()
        os.system("fluidsynth -l -i -a file %s %s -F %s -r 44100" %
                  (Sample.SOUNDFONT, self.mid_filename, self.tmp_filename))

    def transform_wav(self, suffix, start_s=0, duration=1,
                      pitch_shift_base_freq=440, pitch_shift_hz=0,
                      resample_hz=44100, resample_bits=16,
                      volume=1.0, strip_fundamental=False):
        shift_cents = 0
        if pitch_shift_hz != 0:
            shift_cents = 1200.0 * \
                math.log((pitch_shift_base_freq + pitch_shift_hz) /
                         pitch_shift_base_freq, 2)

        self.new_freq = pitch_shift_base_freq + pitch_shift_hz

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
        # Keep only attack, sustain, and decay wav files
        os.remove(self.mid_filename)  # Remove midi file
        os.remove(self.tmp_filename)  # Remove -full .wav file


if __name__ == "__main__":
    sample = Sample("c-minor.mid")
    sample.write_chord(Chord.chord(Chord.Min, Note.note("C4")))
    sample.new_track(19)  # Church organ
    sample.write_chord(Chord.chord(Chord.Min, Note.note("C5")))
    sample.save()

    sample = Sample("notes.mid")
    sample.write_notes(Note.notes(["C4", "D4", "E4"]))
    sample.make_wav()
