import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load prepared data
network_input = pickle.load(open("network_input.pkl", "rb"))
network_output = pickle.load(open("network_output.pkl", "rb"))
pitchnames = pickle.load(open("pitchnames.pkl", "rb"))

n_vocab = len(pitchnames)

# Build model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(n_vocab, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Save best model
checkpoint = ModelCheckpoint(
    "lstm_music_model.h5",
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

# Train
model.fit(
    network_input,
    network_output,
    epochs=20,
    batch_size=64,
    callbacks=[checkpoint]
)

print("✅ Model training complete")
