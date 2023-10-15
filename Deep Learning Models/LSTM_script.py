import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


raw_text = open("data.txt", 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
print(raw_text)

# chars = set(raw_text)  unique characters in the file
chars = sorted(list(set(raw_text)))
# print(chars)

char_to_int = dict((c, i) for i, c in enumerate(chars))
print(char_to_int)


n_chars = len(raw_text)
n_vocabs = len(chars)

print("chars", n_chars, "vocabs", n_vocabs)

seq_length = 15
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i+seq_length]
    # print("seq_in: ", seq_in)
    seq_out = raw_text[i+seq_length]
    # print("seq_out: ", seq_out)
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append([char_to_int[seq_out]])

n_patterns = len(dataY)
# print("Total patterns: ", n_patterns)
print(dataX)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocabs)
print(X)

from tensorflow.keras.utils import to_categorical
y = to_categorical(dataY)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callback_list = [checkpoint]

epochs = 10
batch_size = 128
model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callback_list)


filename = "weights-improvement-10-2.4011.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

start = np.random.randint(0, len(dataX)-1)



length = 10
final = []

# Initialize the pattern variable with a seed sequence
start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]

int_to_char = {i: c for c, i in char_to_int.items()}

for i in range(length):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocabs)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    
    result = int_to_char[index]
    final.append(result)
    pattern.append(index)

    pattern = pattern[1:]

print(''.join(final))
print(pattern)