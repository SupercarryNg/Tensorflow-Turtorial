import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# model = tf.keras.Sequential(
#     [
#         tf.keras.Input(shape=(28, 28, 1)),
#         layers.Conv2D(256, kernel_size=2, padding='valid', activation='relu'),
#         layers.MaxPool2D(),
#         layers.Conv2D(128, kernel_size=2, padding='valid', activation='relu'),
#         layers.MaxPool2D(),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(10, activation='softmax')
#     ]
# )
#
# print(model.summary())


def my_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, use_bias=False)(inputs)  # we wanna use activation function after batch normalization
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


model = my_model()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
