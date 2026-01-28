import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical

SEQUENCE_LENGTH = 100

def extract_notes():
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

notes = extract_notes()
print("Total notes extracted:", len(notes))

# Create mappings
pitchnames = sorted(set(notes))
note_to_int = {note: i for i, note in enumerate(pitchnames)}

# Prepare sequences
network_input = []
network_output = []

for i in range(len(notes) - SEQUENCE_LENGTH):
    seq_in = notes[i:i + SEQUENCE_LENGTH]
    seq_out = notes[i + SEQUENCE_LENGTH]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)
n_vocab = len(pitchnames)

network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
network_input = network_input / float(n_vocab)
network_output = to_categorical(network_output)

# Save everything
pickle.dump(network_input, open("network_input.pkl", "wb"))
pickle.dump(network_output, open("network_output.pkl", "wb"))
pickle.dump(pitchnames, open("pitchnames.pkl", "wb"))

print("✅ Data prepared and saved")
