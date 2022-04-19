from tensorflow import keras as k
from keras.layers import Conv2D, LeakyReLU, Input, Concatenate, Conv2DTranspose
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential as Sequ, load_model
from keras.models import Model as Mod
from dataset import img_size
import tensorflow as tf


def discriminator():
    mod = Sequ()
    mod.add(Conv2D(32, kernel_size=(7, 7), strides=1, activation='relu', input_shape=(120, 120, 3)))
    mod.add(MaxPooling2D())
    mod.add(Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu'))
    mod.add(MaxPooling2D()),
    mod.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'))
    mod.add(MaxPooling2D())
    mod.add(Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'))
    mod.add(MaxPooling2D())
    mod.add(Flatten())
    mod.add(Dense(512, activation='relu'))
    mod.add(Dense(128, activation='relu'))
    mod.add(Dense(16, activation='relu'))
    mod.add(Dense(1, activation='sigmoid'))
    return mod


def generator_model():
    inputs = Input(shape=(img_size, img_size, 1))

    conv1 = Conv2D(16, kernel_size=(5, 5), strides=1)(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=1)(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=1)(conv1)
    conv1 = LeakyReLU()(conv1)

    conv2 = Conv2D(32, kernel_size=(5, 5), strides=1)(conv1)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(64, kernel_size=(3, 3), strides=1)(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(64, kernel_size=(3, 3), strides=1)(conv2)
    conv2 = LeakyReLU()(conv2)

    conv3 = Conv2D(64, kernel_size=(5, 5), strides=1)(conv2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(128, kernel_size=(3, 3), strides=1)(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(128, kernel_size=(3, 3), strides=1)(conv3)
    conv3 = LeakyReLU()(conv3)

    bottleneck = Conv2D(128, kernel_size=(3, 3), strides=1, activation='tanh', padding='same')(conv3)

    concat_1 = Concatenate()([bottleneck, conv3])
    conv_up_3 = Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu')(concat_1)
    conv_up_3 = Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_3)
    conv_up_3 = Conv2DTranspose(64, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_3)

    concat_2 = Concatenate()([conv_up_3, conv2])
    conv_up_2 = Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu')(concat_2)
    conv_up_2 = Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_2)
    conv_up_2 = Conv2DTranspose(32, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_2)

    concat_3 = Concatenate()([conv_up_2, conv1])
    conv_up_1 = Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(concat_3)
    conv_up_1 = Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_1)
    conv_up_1 = Conv2DTranspose(3, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_1)
    mod = Mod(inputs, conv_up_1)
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


gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
# generator = load_model('C:/Users/Gowtham nag/PycharmProjects/Image_colorizer/Trained_generator.h5',compile=False)
discriminator = load_model('C:/Users/Gowtham nag/PycharmProjects/Image_colorizer/Trained_discriminator.h5',compile=False)
# generator = generator_model()
# discriminator = discriminator()

generator.summary()
discriminator.summary()
