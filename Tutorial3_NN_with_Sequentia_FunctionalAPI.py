import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# GPU Test
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(-1, x_train.shape[-2] * x_train.shape[-1]).astype('float32')  # 28*28
x_train = x_train / 255.0
x_test = x_test.reshape(-1, x_test.shape[-2] * x_test.shape[-1]).astype('float32')
x_test = x_test / 255.0

# Sequential API
model = tf.keras.Sequential(

    [
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='relu'),

    ]
)

# Functional API
inputs = tf.keras.Input(shape=784)
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
