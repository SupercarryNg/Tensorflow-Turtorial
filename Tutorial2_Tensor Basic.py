import tensorflow as tf


# Initialization of Tensors
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
print(x)
x = tf.constant([[1, 2, 3], [4, 5, 6]])
print(x)
x = tf.ones((3, 3))
print(x)
x = tf.zeros((2, 3))
print(x)
x = tf.eye(3)
print(x)
x = tf.random.normal((32, 3, 224, 224), mean=0, stddev=1)
print(x)
print(x.shape)
x = tf.random.uniform((1, 3), minval=0, maxval=1)
print(x)
x = tf.range(start=1, limit=10, delta=2)  # [1, 10), step=2
print(x)
x = tf.cast(x, dtype=tf.float64)  # convert type
# tf.float (16, 32, 64), tf.int(8, 16, 32, 64), tf.bool
print(x)

# Mathematical Operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

z = x + y  # tf.add(x, y)
print(z)
z = x - y  # tf.subtract(x, y)
print(z)
z = x / y  # tf.divide(x, y)
print(z)
z = x * y  # tf.multiply(x, y)
print(z)
z = tf.tensordot(x, y, axes=1)  # x1*y1 + x2*y2 + x3*y3
print(z)
x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = x @ y  # matrix multiply
print(z)

# Indexing
x = tf.constant([0, 1, 1, 2, 3, 4, 4])
print(x[:])
print(x[1:])
print(x[1:3])
print(x[::2])
print(x[::-1])

indices = [0, 3]
print(tf.gather(x, indices))  # same as x[indices] -> could only be used in pytorch

# Reshaping
x = tf.range(9)
print(x)
x = tf.reshape(x, (3, 3))  # same as x.reshape(3, 3)
print(x)
print(tf.transpose(x, perm=[1, 0]))



