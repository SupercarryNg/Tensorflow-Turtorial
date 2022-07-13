import tensorflow as tf
from tensorflow.keras import layers, Model, losses, optimizers
from tensorflow.keras.datasets import mnist

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0


class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',  # specify the name for further saving and loading model
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )

        self.b = self.add_weight(
            name='b',
            shape=(self.units, ),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MyModel(Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(num_classes)

    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)

    def model(self):
        x = tf.keras.Input(shape=(28*28))
        return Model(inputs=[x], outputs=self.call(x))


model = MyModel()
model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizers.Adam(),
    metrics=['accuracy']
)
print(model.model().summary())
model.fit(x_train, y_train, batch_size=128, epochs=3, verbose=2)