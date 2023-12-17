import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import numpy as np

# Provided text
text = """The outer most layer of Earth is often referred to as the biosphere. The biosphere is the layer of living organisms that cover the surface of the planet and it readily distinguishes the planet from all others in the solar system ("Biosphere", 2003). Due to its flowing water and oxygen, Earth is the only planet in the galaxy that is naturally habitable by humans. It is estimated that life has existed on Earth for 3.5 to 4 billion years.
The Earth's surface developed much like the other planets in the galaxy, through four major geological processes. The first process is known as Impact Cratering. Impact craters are the result of comets or asteroids impacting with the surface of a planet. Second, is the process known as volcanism. Volcanism is the result of molten rock and lava spewing from the Earth's interior to its surface. Third, is the process of tectonics. Tectonics is the disruption of a planet's surface by internal stresses. Fourth and finally is the process of erosion. Erosion is the result of wearing down or building up geological features by wind, water, and other planetary weather .
Between the crust and mantle of the Earth, a layer of rock known as the lithosphere exists. The lithosphere is a hard layer of rock that is composed of mostly crust but does include a small portion of the mantle. The lithosphere is broken into plates and allows the softer rock, known as the asthenosphere, to move and flow. This process results in the lithosphere gradually moving and "creating the phenomenon known as continental drift" (Bennet, J., et al., p. 230)."""

# Create a set of unique characters in the text
chars = sorted(set(text))

# Create dictionaries for character-to-index and index-to-character mappings
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for idx, char in enumerate(chars)}

# Create sequences of 20 characters and their corresponding next character
seq_length = 20
step = 1
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])

# Create input and output data in the appropriate format
X = np.zeros((len(sequences), seq_length, len(chars)), dtype=bool)
y = np.zeros((len(sequences), len(chars)), dtype=bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

print("X", X)
print("Y", y)
# Build an LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=10, batch_size=128)

# Function to generate text
def generate_text(model, seed_text, next_chars, temperature=1.0):
    generated_text = seed_text
    for _ in range(next_chars):
        x = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(seed_text):
            x[0, t, char_to_index[char]] = 1
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text

# Function to sample the next character based on predicted probabilities
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate the next 5 characters for a 20-character sequence
seed_sequence = "The outer most layer"
next_characters = 5
generated_sequence = generate_text(model, seed_sequence, next_characters, temperature=0.5)
print("Generated Sequence:", generated_sequence)
