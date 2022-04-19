from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import numpy as np
from PIL import Image

############################################################heliochromy
batch_size = 10
img_size = 120
total_trainable_params = 250

master_path = 'data\\color'
chroma_images = []
grey_scaled_images = []


for each_image in os.listdir(master_path)[0: total_trainable_params]:
    image1 = Image.open(os.path.join(master_path, each_image)).resize((img_size, img_size))
    rgb_img_array = (np.asarray(image1)) / 255
    gray_image = image1.convert('L')
    gray_img_array = (np.asarray(gray_image).reshape((img_size, img_size, 1))) / 255
    chroma_images.append(gray_img_array)
    grey_scaled_images.append(rgb_img_array)
train_chroma_images, test_chroma_images, train_grey_scaled_images, test_grey_scaled_images = train_test_split(np.array(chroma_images), np.array(grey_scaled_images), test_size=0.5)
dataset = tf.data.Dataset.from_tensor_slices((train_chroma_images, train_grey_scaled_images))
dataset = dataset.batch(batch_size)
