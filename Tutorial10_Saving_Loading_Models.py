import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model, losses, optimizers
from tensorflow.keras.datasets import mnist

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0


# 1. How to save and load model weights
# 2. Save and load entire model (Serializing Model)
model = Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=128, epochs=3, verbose=2)
model.evaluate(x_test, y_test, batch_size=1, verbose=2)

save_path = 'save_model/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model.save(save_path)
print('model saved')

model = tf.keras.models.load_model(save_path)
model.fit(x_train, y_train, batch_size=128, epochs=3, verbose=2)
model.evaluate(x_test, y_test, batch_size=1, verbose=2)