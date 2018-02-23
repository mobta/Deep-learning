# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 19:21:35 2018

@author: Tahar
"""

from keras.models import Sequential
from keras.layers import MaxPooling2D, Dense, Flatten, Convolution2D

# initialize the CNN
classifier = Sequential()

# step 1 - Convolution 
classifier.add(Convolution2D(32,3, 3, input_shape=(64, 64, 3), activation='relu'))

# step 2 - pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step 3 - flatten
classifier.add(Flatten())

# step 4 - build the ANN
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# step 5 - compile the model
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# step 6 - image augmentation to avoid overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=10,
                         validation_data=test_set,
                         validation_steps=2000)
