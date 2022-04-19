import warnings
from PIL import Image
from dataset import dataset, test_chroma_images,test_grey_scaled_images
import tensorflow as tf
from Modeltwo import generator, discriminator, generator_loss, discriminator_loss, gen_optimizer, disc_optimizer
from matplotlib import pyplot as plt#
import sys
import numpy as np


x_list = []


# y_lmist = []


def traingan(input_x, real_y):
    # y_list.append(real_y.re)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(input_x, training=True)
        real_output = discriminator(real_y, training=True)
        generated_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(generated_images, real_y)
        x_list.append(gen_loss)
        disc_loss = discriminator_loss(real_output, generated_output)
    print('.', end=' ')
    # tf.keras.backend.print_tensor( tf.keras.backend.mean( gen_loss ), gen_loss + disc_loss )

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Optimizing the losses with Adam
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


num_of_epochs = 15

for epoch in range(num_of_epochs):
    print("\n", epoch)
    try:
        for (x, y) in dataset:
            # Here ( x , y ) represents a batch from our training dataset.# print(x.shape)
            traingan(x, y)
    except KeyboardInterrupt:
        discriminator.save("Trained_discriminator2.h5")
        generator.save("Trained_generator2.h5")
        sys.exit(0)

print("\ndone training!!")
discriminator.save("Trained_discriminator.h5")
generator.save("Trained_generator.h5")
y = generator(test_chroma_images[0:]).numpy()

plt.figure()
plt.title('Graph')
plt.plot(x_list)  # , y_list)
plt.show()
print(len(test_chroma_images))

'''
for i in range(5):
    plt.figure(figsize=(20, 10))
    or_image = plt.subplot(3, 3, 1)
    or_image.set_title('Grayscale Input', fontsize=16)
    plt.imshow(test_chroma_images[i].reshape((120, 120)), cmap='gray')

    in_image = plt.subplot(3, 3, 2)
    image = Image.fromarray((y[i] * 255).astype('uint8')).resize((400, 400))
    image = np.asarray(image)
    in_image.set_title('Colorized Output', fontsize=16)
    plt.imshow(image)

    ou_image = plt.subplot(3, 3, 3)
    image = Image.fromarray((test_grey_scaled_images[i] * 255).astype('uint8')).resize((1024, 1024))
    ou_image.set_title('True Image', fontsize=16)
    plt.imshow(image)
    plt.show()'''
