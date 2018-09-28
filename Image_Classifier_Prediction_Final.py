#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 10:58:18 2018

@author: kamran
"""

# Image Classifier using CNN. It will classify Cat and Dog images.

# Import the module 
#import os
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
#from keras.preprocessing.image import load_img
#import matplotlib.pyplot as plt
#import cv2
from PIL import Image
import time


img_width, img_height = 128, 128
time_limit = 5

# Ask for Model and Image
# model = #model = load_model('CNN_cats_dogs_3_hiddenLayes_my_model.h5')
# Get the Model of Cats_Dogs
print("Enter the Path of Model")
model_path = input()
print("\n")
print("Enter the Path of Image")
image_path = input()
#model_path = "/home/kamran/Link to CVM/Image_Classifier/Cat_Dog/CNN_cats_dogs_3_hiddenLayes_earlystop4_my_model.h5"
# Load the Model
print("Loading The Model")
model = load_model(model_path)
# Get the image path

#image_path = "/home/kamran/Link to CVM/Image_Classifier/Udemy/dataset/single_prediction/cat_or_dog_4.jpg"
# Resize the Image
test_image = image.load_img(image_path, target_size = (img_width, img_height))
# Convert the image into array
test_image = image.img_to_array(test_image)
# Add one more dimenssion
test_image = np.expand_dims(test_image, axis = 0)
# Do the prediction
result = model.predict(test_image)

# Display the Class 
#print("{'cats': 0, 'dogs': 1}")

#import Image
time.sleep(time_limit)


try:  
    image1 = Image.open(image_path)
except IOError: 
    pass

"""
# Get the Resolution of the Screen
import subprocess
cmd = ['xrandr']
cmd2 = ['grep', '*']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
p.stdout.close()
 
resolution_string, junk = p2.communicate()
resolution = resolution_string.split()[0]
resolution = resolution.decode("utf-8") 
width = int(resolution.split("x")[0].strip())
heigth = int(resolution.split("x")[1].strip())
"""


# Show the Image 
width1, height1 = image1.size
if width1 <= 500:
    image1.show()
else:
    image1 = image1.resize((int(width1/3), int(height1/3)))
    image1.show()

#!pip install Image

"""
img = cv2.imread(image_path,0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import ctypes
user32 = ctypes.
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)



img = load_img(image_path)
plt.imshow(img)
"""

# Display the Prediction
if result[0][0] == 1:
    prediction = 'This is  a dog'
else:
    prediction = 'This is a cat'

print(prediction)





