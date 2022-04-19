from tkinter import Image
import os

import numpy as np

from dataset import chroma_images

chrom_images = []
gry_scaled_images = []

img_size = 120

master_path=input(("Enter the directory of ur image...:"))
image1 = Image.open(os.path.join(master_path)).resize((img_size, img_size))
rgb_img_array = (np.asarray(image1)) / 255
gray_image = image1.convert('L')
gray_img_array = (np.asarray(gray_image).reshape((img_size, img_size, 1))) / 255
chrom_images.append(gray_img_array)
gry_scaled_images.append(rgb_img_array)