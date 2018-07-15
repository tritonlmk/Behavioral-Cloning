# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 16:37:36 2018

@author: linmingkun
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense,Flatten,Lambda,Activation,Dropout
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D

import socketio
import argparse
import os
import csv
import cv2
import itertools
import json

def load_data():
    'load/split and return data'
    'There is a problem that simply use for line in reader, it will import the first line that is not the data we want to use'
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in itertools.islice(reader,1,None):
            samples.append(line)
    shuffle(samples)
    train_samples, valid_samples = train_test_split(samples, test_size=0.2)
    return train_samples, valid_samples

def generator(samples, batch_size, adjust_para):
    'generator data without out of memory'
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            steering = []
            
            for batch_sample in batch_samples:
                'randomly choose picture from center, left and right'
                
                name_center = './Data/IMG/'+batch_sample[0].split('/')[-1]
                current_steering_center = float(batch_sample[3])
                name_left = './Data/IMG/'+batch_sample[1].split('/')[-1]
                current_steering_left = current_steering_center + adjust_para
                name_right = './Data/IMG/'+batch_sample[2].split('/')[-1]
                current_steering_right = current_steering_center - adjust_para
                
                current_image_center = cv2.imread(name_center)
                current_image_left = cv2.imread(name_left)
                current_image_right = cv2.imread(name_right)
#                current_image, current_steering = preprocess_image(current_image, current_steering)
                images.append(current_image_center)
                images.append(current_image_left)
                images.append(current_image_right)
                steering.append(current_steering_center)
                steering.append(current_steering_left)
                steering.append(current_steering_right)
                
            X_train = np.array(images)
            Y_train = np.array(steering)
            yield shuffle(X_train, Y_train)
            

def preprocess_image(image,steering):
    'crop the image， for visualization, real cropping in done in model'
#    image = image[50:-23,:,:]
    'resize the image'
    
    'convert the image from rgb to yuv'
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    'random flip the image'
    if np.random.rand() < 0.5:
        image = cv2.flip(image,1)
        steering = -steering;
    return image,steering

def image_out(image_sets):
    'read the image sets and output one image'
    name = './Data/IMG/'+image_sets[0][0].split('/')[-1]
    image = cv2.imread(name)
    return image
    

def NV_Network(model_image_shape, keep_prob):
    'Model from NVIDIA e2e-dl-using-px wihtout any change'
    model = Sequential()
    
    model.add(Lambda(lambda x:x/255-0.5, input_shape=model_image_shape))
    model.add(Cropping2D(cropping=((50,21), (0,0)), input_shape=model_image_shape))
    model.add(Convolution2D(24,5,5, border_mode='same', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, border_mode='same', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, border_mode='same', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, border_mode='same', subsample=(1,1), activation='relu'))
    model.add(Convolution2D(64,3,3, border_mode='same', subsample=(1,1), activation='relu'))
    
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    
    model.add(Dense(1164,activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    
    return model


def train_e2e(model, train_samples_in, valid_samples_in, nb_epochs, nb_train_step, nb_valid_step):
    'Train the model using data from generator'
    'using generator and model.fit_generator to samve memory'
    
    train_generator = generator(train_samples_in, 128, 0.2)
    validation_generator = generator(valid_samples_in, 128, 0.2)
#    picture_shape = train_generator[0].shape
    
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(generator = train_generator,
                        steps_per_epoch = nb_train_step,
                        validation_data = validation_generator,
                        validation_steps = nb_valid_step,
                        epochs = nb_epochs,
                        verbose = 2)
    
    model.save('model.h5')
    print('saved model weight')
    model_json = model.to_json()
    with open('model.json', 'w') as file:
        file.write(model_json)
        
    return history_object


train_samples, valid_samples = load_data()
print("number of training samples is:", len(train_samples))
print("number of validation samples is:", len(valid_samples))

image_sample = image_out(train_samples)
image_after, not_used = preprocess_image(image_sample,0)
input_image_shape = image_after.shape
plt.subplot(2,1,1)
image_after = image_after[50:-21,:,:]
plt.imshow(image_after)
plt.axis('off')

model_test = NV_Network(input_image_shape, 0.5)
history_object_out = train_e2e(model_test, train_samples, valid_samples, 3, 140, 140)

print(history_object_out.history.keys())

plt.subplot(2,1,2)
plt.plot(history_object_out.history['loss'])
plt.plot(history_object_out.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squareed error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


print('end')