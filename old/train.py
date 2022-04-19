from PIL import Image
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import Model
import Modeltwo
from Modeltwo import generator as gen
from Modeltwo import generator, generator_loss, discriminator, discriminator_loss
from dat import test_y
from dat import test_x, img_size
from dat import dataset
import sys


###########################################


def train_step(x, y):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # print(x.shape)print(y.shape)
        generated_images = generator(x, training=True)
        # Probability that the given image is real -> D( x )
        real_output = discriminator(y, training=True)
        # Probability that the given image is the one generated -> D( G( x ) )
        # print("hai",generated_images.shape)
        generated_output = discriminator(generated_images, training=True)

        # L2 Loss -> || y - G(x) ||^2
        gen_loss = generator_loss(generated_images, y)
        # Log loss for the discriminator
        disc_loss = discriminator_loss(real_output, generated_output)
    print(".", end="")
    # print("gen-loss,(gen+disc-losses)=",end="")
    # tf.keras.backend.print_tensor(tf.keras.backend.mean(gen_loss),gen_loss   + disc_loss)
    # tf.keras.backend.print_tensor(gen_loss + disc_loss)

    # Compute the gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Optimize with Adam
    Modeltwo.gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    Modeltwo.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


num_of_epochs = 15
# (dataset)+
print("Starting the networks!!!")
for e in range(num_of_epochs):
    print("\nno of epoch:", e)
    try:
        for (x, y) in dataset:
            # Here ( x , y ) represents a batch from our training dataset.
            # print(x.shape)
            train_step(x, y)
            continue
    except KeyboardInterrupt:
        s = input("\nQuitting... \nDo u wish to save the models?[Y/n]:")
        if s == 'Y':
            discriminator.save("Trained_discriminator.h5")
            generator.save("Trained_generator.h5")
            print("Models saved!!!")
            sys.exit(0)
        #discriminator.save("Trained_discriminator.h5")
        #generator.save("Trained_generator.h5")
        sys.exit(0)
discriminator.save("Trained_discriminator.h5")
generator.save("Trained_generator.h5")
print("Done Training!!")
# Results
y = generator(test_x[0:]).numpy()

for i in range(len(test_x)):
    plt.figure(figsize=(10, 10))
    or_image = plt.subplot(3, 3, 1)
    or_image.set_title('Grayscale Input', fontsize=16)
    plt.imshow(test_x[i], cmap='gray')  # .reshape((120, 120,1)), cmap='gray')

    in_image = plt.subplot(3, 3, 2)
    image = Image.fromarray((y[i] * 255).astype('uint8')).resize((1024, 1024))
    image = np.asarray(image)
    in_image.set_title('Colorized Output', fontsize=16)
    plt.imshow(image)

    ou_image = plt.subplot(3, 3, 3)
    image = Image.fromarray((test_x[i] * 255).astype('uint8')).resize((1024, 1024))
    ou_image.set_title('Ground Truth', fontsize=16)
    plt.imshow(image)

    plt.show()
