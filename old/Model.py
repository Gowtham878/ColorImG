from tensorflow import keras as k
import tensorflow as tf
from keras.models import Sequential as Sequ
from keras.layers import Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.layers import MaxPool2D
from keras.layers import LeakyReLU
import os


def generator():
    model = Sequ()
    model.add(Input(shape=(120, 120, 1),name="Input"))
    model.add(Conv2D(128, kernel_size=(5, 5), strides=1, ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    '''model.add(Conv2D(64, kernel_size=(3, 3), strides=1))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))'''
    model.add(Conv2D(32, kernel_size=(3, 3), strides=1))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='tanh', strides=1, padding='same'))  # bottleneck
    model.add(Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', strides=1))  # curveup
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    '''model.add(Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', strides=1))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))'''
    model.add(Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', strides=1))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2DTranspose(3, kernel_size=(5, 5), activation='relu', strides=1))
    return model


def discriminator():
    mod = Sequ()
    mod.add(Conv2D(32, kernel_size=(7, 7), strides=1, activation='relu', input_shape=(120, 120, 3)))
    mod.add(MaxPool2D())
    mod.add(Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu'))
    mod.add(MaxPool2D()),
    mod.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'))
    mod.add(MaxPool2D())
    mod.add(Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'))
    mod.add(MaxPool2D())
    mod.add(Flatten())
    mod.add(Dense(512, activation='relu'))
    mod.add(Dense(128, activation='relu'))
    mod.add(Dense(16, activation='relu'))
    mod.add(Dense(1, activation='sigmoid'))
    return mod


cross_entropy = k.losses.BinaryCrossentropy()
mse = k.losses.MeanSquaredError()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape, maxval=0.1),
                              real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape, maxval=0.1),
                              fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, real_y):
    real_y = tf.cast(real_y, 'float32')
    return mse(fake_output, real_y)


gen_optimizer = tf.keras.optimizers.Adam(lr=0.0005)
disc_optimizer = tf.keras.optimizers.Adam(lr=0.0005)
generator = generator()
discriminator = discriminator()
generator.summary()
print("###################################################")
# discriminator.summary()
