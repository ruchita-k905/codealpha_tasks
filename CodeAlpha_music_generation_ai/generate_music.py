import pickle
import numpy as np
import random
from music21 import note, chord, stream, instrument
from tensorflow.keras.models import load_model

SEQUENCE_LENGTH = 100

# Load data and model
network_input = pickle.load(open("network_input.pkl", "rb"))
pitchnames = pickle.load(open("pitchnames.pkl", "rb"))
model = load_model("lstm_music_model.h5")

int_to_note = {i: note for i, note in enumerate(pitchnames)}
n_vocab = len(pitchnames)

# Pick a random seed
start = random.randint(0, len(network_input) - 1)
pattern = network_input[start]

generated_notes = []

for _ in range(200):
    prediction_input = np.reshape(pattern, (1, SEQUENCE_LENGTH, 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)

    result = int_to_note[index]
    generated_notes.append(result)

    pattern = np.append(pattern, [[index]], axis=0)
    pattern = pattern[1:]

# Convert to MIDI
offset = 0
output_notes = []

for pattern in generated_notes:
    if '.' in pattern:
        chord_notes = pattern.split('.')
        notes_list = [note.Note(int(n)) for n in chord_notes]
        new_chord = chord.Chord(notes_list)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music_lstm.mid')

print("🎵 Music generated: generated_music_lstm.mid")
