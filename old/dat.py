from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
import cv2

##################################################################################

batch_size = 2

image_size = 120
img_size = 120
dataset_split = 10
x = [] # gray images
y = [] # color images
master_dir = 'data/color'
clr_dir = 'data/color'
gry_dir = 'train_black'
# gry_dir ='data/color'

#################################################################################
print("Converting Images...")
for image_file in os.listdir(master_dir)[0: dataset_split]:
    rgb_image = Image.open(os.path.join(master_dir, image_file)).resize((img_size, img_size))
    im = np.float32(rgb_image)
    rgb_img_array = (np.asarray(im)) / 255
    # print(rgb_img_array.shape)
    y.append(rgb_img_array)
    gray_image = rgb_image.convert('L')
    gray__image=np.float32(gray_image)
    gray_img_array = (np.asarray(gray__image).reshape((img_size, img_size, 1))) / 255
    x.append(gray_img_array)
    # print('hao')
'''print("preparing data set for discriminator")
for image in os.listdir(clr_dir)[0:2500]:  # rgbforD
    rgb_image = Image.open(os.path.join(clr_dir, image)).resize((img_size, img_size))
    rgb_image_arr = (np.asarray(rgb_image)) / 255
    x.append(rgb_image_arr)
print("preparing data set for generator")

for img in os.listdir(clr_dir)[0:1000]:  # greyforG
    gr_image = np.array(Image.open(os.path.join(clr_dir, img)).resize((img_size, img_size))) / 255
    gr_image = np.float32(gr_image)
    gr_image_m = cv2.cvtColor(gr_image, cv2.COLOR_BGRA2GRAY)
    gr_image_m = gr_image_m
    y.append(gr_image_m)
for image in os.listdir(gry_dir)[0:dataset_split]:
    gr_image = np.asarray(Image.open(os.path.join(gry_dir, image)).resize((img_size, img_size))) / 255
    gr_image = np.float32(gr_image)
    gr_image_m = cv2.cvtColor(gr_image, cv2.COLOR_BGRA2GRAY)
    gr_image_m = gr_image_m.reshape(img_size,img_size,1)
    x.append(gr_image_m)'''
# print(len(y))
# x=np.array(x)
# print("x",x.shape)
print("Done preparing datasets!!!\n")
# split
train_x, test_x, train_y, test_y = train_test_split( np.array(x) , np.array(y) , test_size=0.1 )

# print("y",train_y.shape)
print("Test split done")
# Construct tf.data.Dataset object
#print(type(train_x), "\n", type(train_y))
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(batch_size)
print(dataset)

''''from PIL import Image
import cv2
import numpy as np
im=np.array(Image.open('C:\\Users\\Gowtham nag\\PycharmProjects\\
Image_colorizer\\data\\train_black\\image0000.jpg'))/255
im = np.float32(im)
m= cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
print(m.shape)'''
