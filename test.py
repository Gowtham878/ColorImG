import matplotlib.pyplot as plt
from keras.models import load_model
#from Modeltwo import generator
from dataset import test_chroma_images, test_grey_scaled_images
from PIL import Image
import numpy as np

generator = load_model('C:/Users/Gowtham nag/PycharmProjects/Image_colorizer/Trained_generator.h5',compile=False)

y = generator(test_chroma_images[0:15]).numpy()

for i in range(15):
    plt.figure(figsize=(10, 10))
    or_image = plt.subplot(1, 3, 1)
    or_image.set_title('Grayscale Input', fontsize=16)
    plt.imshow(test_chroma_images[i].reshape((120, 120)), cmap='gray')

    in_image = plt.subplot(1, 3, 2)
    image = Image.fromarray((y[i] * 255).astype('uint8')).resize((1024, 1024))
    image = np.asarray(image)
    in_image.set_title('Colorized Output', fontsize=16)
    plt.imshow(image)

    ou_image = plt.subplot(1, 3, 3)
    image = Image.fromarray((test_grey_scaled_images[i] * 255).astype('uint8')).resize((1024, 1024))
    ou_image.set_title('True Image', fontsize=16)
    plt.imshow(image)
    plt.show()
