# Music theory primitives
# Mohit Cheppudira <mohit@muthanna.com>

module Music

# Notes
module N
  Values = {
    "c" => 0,
    "d" => 2,
    "e" => 4,
    "f" => 5,
    "g" => 7,
    "a" => 9,
    "b" => 11
  }

  Names = ["C", "Cs", "D", "Eb", "E", "F", "Fs", "G", "Ab", "A", "Bb", "B"]

  # Takes a note in the form "C#4", "Bb2", "A5", etc. and returns
  # the MIDI note number.
  def note(str)
    if str =~ /^([ABCDEFGabcdefg])([b#]?)([0-9])$/
      octave = $3.to_i
      note = $1.downcase
      acc = $2.downcase

      shift = 0
      if acc == "b"
        shift -= 1
      elsif acc == '#'
        shift += 1
      end

      value = ((octave+1) * 12) + Values[note] + shift
      return value.to_i
    end
  end

  def midi_note(n, o)
    return ((o+1) * 12) + n
  end

  module_function :note, :midi_note
end

# Intervals
module I
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
end

# Chords
module C
  include I

  Maj = [IM3, Ip5]
  Min = [Im3, Ip5]
  Dim = [Im3, Id5]
  Aug = [IM3, Ia5]
  Sus2 = [IM2, Ip5]
  Sus4 = [Ip4, Ip5]
  P5 = [Ip5]
  Maj7 = Maj << IM7
  Dom7 = Maj << Im7
  Min7 = Min << Im7
  MinMaj7 = Min << IM7
  Maj9 = Maj7 << I9
  Maj7s11 = Maj7 << Is11
  Dom7s11 = Dom7 << Is11
  Dom7add13 = Dom7 << I13
  Maj13 = Maj9 << I13
  Min9 = Min7 << I9
  Min11 = Min9 << I11
  Min13 = Min11 << I13
  Dom7s5 = [IM3, Ia5, Im7]
  Dom7b5 = [IM3, Id5, Im7]
  Dom7s5b9 = Dom7s5 << Ib9
  Dom7s5s9 = Dom7s5 << Is9
  Dom7b5b9 = Dom7b5 << Ib9
  Dom7b5s9 = Dom7b5 << Is9

  Names = {
    "Maj" => {tones: Maj, label: "Maj"},
    "Min" => {tones: Min, label: "Min"},
    "Dim" => {tones: Dim, label: "Dim"},
    "Aug" => {tones: Aug, label: "Aug"},
    "Sus2" => {tones: Sus2, label: "Sus"},
    "Sus4" => {tones: Sus4, label: "Sus"},
    "P5" => {tones: Maj, label: "P5"},
    "Maj7" => {tones: Maj7, label: "Maj"},
    "Dom7" => {tones: Dom7, label: "Dom"},
    "Min7" => {tones: Min7, label: "Min"},
    "MinMaj7" => {tones: MinMaj7, label: "Min"},
    "Maj9" => {tones: Maj9, label: "Maj"},
    "Maj7s11" => {tones: Maj7s11, label: "Maj"},
    "Dom7s11" => {tones: Dom7s11, label: "Dom"},
    "Dom7add13" => {tones: Dom7add13, label: "Dom"},
    "Maj13" => {tones: Maj13, label: "Maj"},
    "Min9" => {tones: Min9, label: "Min"},
    "Min11" => {tones: Min11, label: "Min"},
    "Min13" => {tones: Min13, label: "Min"},
    "Dom7s5" => {tones: Dom7s5, label: "Dom"},
    "Dom7b5" => {tones: Dom7b5, label: "Dom"},
    "Dom7s5b9" => {tones: Dom7s5b9, label: "Dom"},
    "Dom7s5s9" => {tones: Dom7s5s9, label: "Dom"},
    "Dom7b5b9" => {tones: Dom7b5b9, label: "Dom"},
    "Dom7b5s9" => {tones: Dom7b5s9, label: "Dom"}
  }

  def gen_chord(intervals, root)
    return [root] + intervals.map {|i| root + i}
  end

  module_function :gen_chord
end

end #music