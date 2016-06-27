#! /usr/bin/env ruby
# Copyright Mohit Cheppudira 2014 <mohit@muthanna.com>
#
# usage: gen_midi_samples.rb [directory]
#
# This script generates audio spectrograms for training the
# deep convnet with synthetic data.

$LOAD_PATH[0, 0] = File.join(File.dirname(__FILE__), '..', 'lib')

require 'midilib/sequence'
require 'midilib/consts'
require 'rake'
require_relative 'music'

include MIDI
include Music

$options = {
  do_singles: true,          # Generate single tones
  do_random_midi: false,      # Generate random MIDI samples
  do_chord_midi: false,       # Generate chord MIDI samples
  do_conversion: true,       # Convert MIDI to WAV and PNG
  generate_labels: true,      # Generate training labels

  num_random_samples: 1000,
  random_sample_groups: [1, 2, 3], # generate samples with 1, 2, and 3 tones.
  num_chord_samples: 100,
  tempo: 120,
  velocity_min: 100,
  velocity_max: 127,
  soundfont: "./soundfont.sf2",
  tmp_dir: "data/tmp",
  midi_dir: "data/midi",
  wav_dir: "data/wav",
  png_dir: "data/png",
  root_labels_file: "data/root_labels.txt",
  chord_labels_file: "data/chord_labels.txt",
  labels_file: "data/labels.txt"
}

# GM_PATCH_NAMES.each_with_index {|p,i| puts "#{i}: #{p}"}
PATCHES = [
  0, # Acoustic Grand Piano
  3, # Honky-tonk Piano
  6, # Harpsichord
  11, # Vibraphone
  16, # Drawbar Organ
  19, # Church Organ
  22, # Harmonica
  24, # Acoustic Guitar (nylon)
  26, # Electric Guitar (jazz)
  30, # Distortion Guitar
  32, # Acoustic Bass
  33, # Electric Bass (finger)
  40, # Violin,
  42, # Cello
  48, # String Ensemble 1
  51, # SynthStrings 2
  52, # Choir Aahs
  56, # Trumpet
  57, # Trombone
  61, # Brass Section
  65, # Alto Sax
  66, # Tenor Sax
  71, # Clarinet
  73, # Flute
  78 # Whistle
]

# Build MIDI Sequence with multiple tracks.
def create_multi_seq(programs, tempo, bend=0)
  seq = Sequence.new()
  track = Track.new(seq)
  seq.tracks << track
  track.events << Tempo.new(Tempo.bpm_to_mpq(tempo))
  track.events << MetaEvent.new(META_SEQ_NAME, "Random Sample")

  programs.each do |program|
    name = GM_PATCH_NAMES[program]
    track = Track.new(seq)
    seq.tracks << track
    track.name = name
    track.instrument = name

    # Add a volume controller event (optional).
    track.events << Controller.new(0, CC_VOLUME, 127)
    track.events << ProgramChange.new(0, program, 0)
    if bend != 0 then
      # 0 - 16383. Default (center): 8192
      track.events << PitchBend.new(0, bend, 0)
    end
  end

  return seq
end

# Take each note in `notes` and put it in a separate track in `seq`
def add_chord_to_multi_seq(seq, notes, duration, velocities)
  if (seq.tracks.length - 1) != notes.length or notes.length != velocities.length then
    raise "'seq.tracks', 'notes', and 'velocities' must have same length."
  end

  notes.each_with_index do |note, i|
    seq.tracks[i + 1].events << NoteOn.new(0, note, velocities[i], 0)
  end

  notes.each_with_index do |note, i|
    seq.tracks[i + 1].events << NoteOff.new(0, note, velocities[i], duration)
  end
end

num_random_samples = $options[:num_random_samples]
velocity_min = $options[:velocity_min]
velocity_max = $options[:velocity_max]
tempo = $options[:tempo]
midi_dir = $options[:midi_dir]

if $options[:do_singles] then
  puts "Generating single tones..."
  labels = File.open(midi_dir + "/singles.txt", "w")

  octaves = [3, 4, 5, 6]
  notes = (0..11)
  bends = [0, 6000, 7000, 7500, 7700, 8500, 9000, 9500, 10000]

  total_midi = octaves.size * notes.size * PATCHES.size * bends.size
  total_wav = total_midi * 2 # attack, sustain
  examples_per_note = PATCHES.size * bends.size

  count = 0
  octaves.each do |o|
    puts "Generating octave: #{o}" 
    notes.each do |n|
      value = N.midi_note(n, o)
      chord = [value]
      velocities = [rand(velocity_max - velocity_min) + velocity_min]

      PATCHES.each do |p| # PATCHES.each
        # pitch-bend (8192 = center, 0 = ignore)
        bends.each do |bend|
          programs = [p]
          seq = create_multi_seq(programs, tempo, bend)
          duration = seq.note_to_delta('half')
          add_chord_to_multi_seq(seq, chord, duration, velocities)
          count += 1
          filename = "singles-V#{value}-P#{p}-O#{o}-N#{n}-B#{bend}.mid"
          puts "Sample #{count}: #{filename}"
          File.open(midi_dir + "/" + filename, "wb") {|file| seq.write(file)}
          labels << filename << " note:#{n} octave:#{o} value:#{value} bend:#{bend}" << "\n"
        end
      end
    end
  end

  labels.close()
  puts "Generated: #{total_midi} MIDI files, #{examples_per_note} files per note."
end

if $options[:do_random_midi] then
  puts "Generating #{num_random_samples} random samples."
  count = 0

  num_random_samples.times do
    bottom = rand(3)
    top = rand(5) + 5
    num_notes = top - bottom
    notes = (0..11).to_a.shuffle.first(num_notes)
    chord = []
    velocities = []

    filename = "random"
    notes.each_with_index do |note, index|
      octave = index + bottom + 1
      velocities << rand(velocity_max - velocity_min) + velocity_min
      value = N.midi_note(note, octave)
      chord << value

      # Add values to filename for training labels
      filename << "-V#{value}"
    end

    programs = PATCHES.shuffle.first(num_notes)
    seq = create_multi_seq(programs, tempo)
    duration = seq.note_to_delta('half')
    add_chord_to_multi_seq(seq, chord, duration, velocities)
    filename << ".mid"

    count += 1
    puts "Sample #{count}: #{filename}"
    File.open(midi_dir + "/" + filename, "wb") {|file| seq.write(file)}
  end
end

if $options[:do_chord_midi] then
  bass_octaves = [0,1,2]
  chord_octaves = [3,4,5]
  lead_octaves = [5,6,7]
  num_chord_samples = $options[:num_chord_samples]

  total = num_chord_samples * C::Names.length
  count = 0

  num_chord_samples.times do
    C::Names.each do |name, chord|
      count += 1
      puts "Generating chord #{count} of #{total}..."

      bass_o = bass_octaves[rand(bass_octaves.length)]
      lead_o = lead_octaves[rand(lead_octaves.length)]

      bass_note = rand(12)
      chord_tones = C.gen_chord(chord[:tones], bass_note)
      tone_pool = chord_tones.shuffle + chord_tones.shuffle + chord_tones.shuffle

      final_tones = [N.midi_note(bass_note, bass_o)]

      # If you change chord_octaves, update the midi_note selection in the next line
      final_tones += tone_pool.first(chord_octaves.length * 2).each_with_index.map {|t, i| N.midi_note(t, 3 + (i%3))}
      final_tones << N.midi_note(rand(12), lead_o) # completely random lead note
      tone_string = final_tones.inject("") {|o, v| o << "V#{v}"}

      programs = PATCHES.shuffle.first(final_tones.length)
      seq = create_multi_seq(programs, tempo)
      duration = seq.note_to_delta('half')
      velocities = final_tones.map {rand(velocity_max - velocity_min) + velocity_min}
      add_chord_to_multi_seq(seq, final_tones, duration, velocities)
      filename = "chord-N#{name}-L#{chord[:label]}-K#{N::Names[bass_note]}#{bass_o}-T#{tone_string}.mid"
      puts "Writing: #{filename} -- #{final_tones.map {|n| Utils.note_to_s(n)}}"
      File.open(midi_dir + "/" + filename, "wb") {|file| seq.write(file)}
    end
  end
end

def gen_spectrograms(raw_file, basename, start_s, end_s, suffix)
  wav_file = $options[:wav_dir] + "/" + basename + "-#{suffix}.wav"
  noise_wav_file = $options[:wav_dir] + "/" + basename + "-noisy-#{suffix}.wav"
  png_file = $options[:png_dir] + "/" + basename + "-#{suffix}.png"
  noise_png_file = $options[:png_dir] + "/" + basename + "-noisy-#{suffix}.png"

  sh "sox -t raw -r 44100 -e signed -b 16 -c 2 #{raw_file} #{wav_file} norm -3 remix 2 " +
       "trim #{start_s} #{end_s} spectrogram -x 330 -y 330 -m -q 128 -r -o #{png_file}"
  # sh "sox #{wav_file} -p synth whitenoise vol 0.1 | sox -m #{wav_file} - #{noise_wav_file}"
  # sh "sox #{noise_wav_file} -n spectrogram -x 330 -y 330 -m -q 128 -r -o #{noise_png_file}"
end

if $options[:do_conversion] then
  tmp_file = $options[:tmp_dir] + "/temp.raw"
  wav_dir = $options[:wav_dir]
  png_dir = $options[:png_dir]

  files = Dir.glob(midi_dir + "/*.mid")
  total = files.length
  files.each_with_index do |file, i|
    puts "\nConverting file #{i} of #{total}..."
    basename = File.basename(file, ".mid")
    sh "fluidsynth -l -i -a file #{$options[:soundfont]} #{file} -F #{tmp_file}"

    # Note 0.33 splits evenly into 330 pixels in the spectrogram
    gen_spectrograms(tmp_file, basename, 0, 0.33, "attack")
    gen_spectrograms(tmp_file, basename, 0.33, 0.33, "sustain")
  end
end

if $options[:generate_labels] then
  png_dir = $options[:png_dir]

  files = Dir.glob(png_dir + "/*.png")
  total = files.length

  roots_output = []
  chords_output = []
  root_labels = {}
  chord_labels = {}
  label_int = 0

  roots_f = File.open($options[:root_labels_file], "w")
  chords_f = File.open($options[:chord_labels_file], "w")
  prefix = Dir.pwd

  files.each_with_index do |file, i|
    # "chord-N#{name}-L#{chord[:label]}-K#{N::Names[bass_note]}#{bass_o}-T#{tone_string}.mid"
    if (file =~ /chord-([^.]+)\.png/) then
      labels = $1.split("-")
      labels.each do |label|
        type = label[0]
        value = label[1..-1]
        case type
        when "K"
          root = value[0..-2]
          if not root_labels[root] then
            root_labels[root] = label_int
            label_int += 1
          end
          roots_f.write("#{prefix}/#{file} #{root_labels[root]}\n")
        when "L"
          if not chord_labels[value] then
            chord_labels[value] = label_int
            label_int += 1
          end
          chords_f.write("#{prefix}/#{file} #{chord_labels[value]}\n")
        when "N"
          puts "Generating label #{i} of #{total}: #{value}"
        end
      end
    end
  end

  roots_f.close
  chords_f.close

  labels_f = File.open($options[:labels_file], "w")
  root_labels.each {|k, v| labels_f.write("#{k}=#{v}\n")}
  chord_labels.each {|k, v| labels_f.write("#{k}=#{v}\n")}
  labels_f.close
end

