import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# RNN
# model = tf.keras.Sequential()
# model.add(tf.keras.Input(shape=(None, 28)))
# model.add(layers.SimpleRNN(512, return_sequences=True, activation='relu'))  # (None, None, 512) -> (bs, hidden, nodes)
# model.add(layers.SimpleRNN(512, activation='relu'))  # No return sequences here, (None, 512) -> (bs, nodes)
# model.add(layers.Dense(10, activation='softmax'))

# GRU
model = tf.keras.Sequential()
model.add(layers.InputLayer(shape=(None, 28)))
model.add(layers.GRU(256, return_sequences=True, activation='relu'))  # (None, None, 512) -> (bs, hidden, nodes)
model.add(layers.GRU(256, activation='relu'))  # No return sequences here, (None, 512) -> (bs, nodes)
model.add(layers.Dense(10, activation='softmax'))

# LSTM
# model = tf.keras.Sequential()
# model.add(layers.InputLayer(shape=(None, 28)))
# model.add(
#     layers.Bidirectional(
#         layers.LSTM(256, return_sequences=True, activation='relu')
#     )
# )  # (None, None, 512) -> (bs, hidden, nodes)
# model.add(
#     layers.Bidirectional(
#         layers.LSTM(256, activation='relu')
#     )
# )  # No return sequences here, (None, 512) -> (bs, nodes)
# model.add(layers.Dense(10, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-04),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=1, verbose=2)