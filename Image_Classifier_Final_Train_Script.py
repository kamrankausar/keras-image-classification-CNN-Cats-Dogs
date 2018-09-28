#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:06:30 2018

@author: kamran
"""

# Part 1 - Loading the Module
print("Loading the required Module")
# Importing the Keras libraries and packages
from keras.models import Sequential # It is use to initilized the 
# Neural Network. Twp ways to initilized the NN 1. Graph or sequence pf 
# Layer, CNN is the sequence of layer so Sequence package 
# Tensorflow is little faster 
#from keras.layers import Convolution2D # To add the Convolutional
from keras.layers import Conv2D
# layer and Images are as input and is 2D so use 2D
from keras.layers import MaxPooling2D # Pooling step to add the pooling layer
from keras.layers import Flatten # Flatten the input from the previous layer
# Then it will become the input of fully connected layers
from keras.layers import Dense # To add the fully connected layer in NN
from keras.layers import Dropout # To prevent the overfitting
from keras.preprocessing.image import ImageDataGenerator # To Load the Images in Batch mode and do some transformation of the Images
from keras.callbacks import EarlyStopping # For Early Stopping of the Model, i.e for consecutive step if val_loss is same then stop training
#import h5py
import datetime # To get the Date and Time
import os


# Change the Directory to Training and Test Folder of the Cats and Dogs Images
# Location of the Training and Test Images
path_train_test  = "/home/kamran/Link to CVM/Image_Classifier/Cat_Dog" 
os.chdir(path_train_test)
# Training and Test Folder
train_set_dir = 'training_set'
test_set_dir = 'test_set'

# To resize the Image 
img_width, img_height = 128, 128

# Early Stopping int
early_stop_value = 4

# Path of the image to test the single image
path_test_image = '/home/kamran/Link to CVM/Image_Classifier/Udemy/dataset/single_prediction/cat_or_dog_19.jpg'

print("Building the Model")

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (img_width, img_height, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

print("Compiling of the Model Starts")
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# Note to show the Probability of the each class see below 
"""
The problem is that you are using the 'sparse_categorical_crossentropy' loss with class_mode='binary' in your ImageDataGenerator.

You have two possibilities here:

    Change the loss to 'categorical_crossentropy' Line No - 72 and set class_mode='categorical' Line Number - 98 and 102.
    Leave the loss as is but set class_mode='sparse'.

"""

print("Getting the Train and Test Data set using the batch mode")
training_set = train_datagen.flow_from_directory(train_set_dir,
                                                 target_size = (img_width, img_height),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory(test_set_dir,
                                            target_size = (img_width, img_height),
                                            batch_size = 32,
                                            class_mode = 'binary')
### settings for keras early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_value, mode='auto')

print("Training of Model Starts...")
classifier.fit_generator(training_set,
                         steps_per_epoch = 12000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000,
                         callbacks=[early_stopping])


# Get the Date and Time
now = datetime.datetime.now()
# Change the Model Name append by Date Hour and Minute 
model_name = 'CNN_cats_dogs_3_hiddenLayes_earlystop_4' + "_" + str(now.day) + "_" + str(now.hour) + "_"  + str(now.minute) + ".h5"
# Save the trained model
classifier.save(model_name)


# Do some testing on given Image
import numpy as np
from keras.preprocessing import image # To load the Image
# Load the Image
test_image = image.load_img(path_test_image, target_size = (img_width, img_height))
# Convert the Image into Array
test_image = image.img_to_array(test_image)
# Add one more dimenssion
test_image = np.expand_dims(test_image, axis = 0)
# Call the model and do prediction
result = classifier.predict(test_image)
# Display the class of the Labels
print(training_set.class_indices)

# Display the result 
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
