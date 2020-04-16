@author: aza8223
"""
###############################################################################
###############################################################################
###############################################################################
"""Deep NN - Project1 - Azad Almasov"""
###############################################################################
###############################################################################
###############################################################################




"""Importing models to be used"""

from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
import pandas as pd
from keras.utils.np_utils import to_categorical

"""Feature extraction function for Transfer learning"""
datagen = ImageDataGenerator(rescale=1./255)
def extract_features(directory, sample_count):
        features = np.zeros(shape=(sample_count, 4, 4, 512))
        labels = np.zeros(shape=(sample_count, len(total_labels)))
    
        generator = datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical')
        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = conv_base.predict(inputs_batch)
            features[i * batch_size : (i + 1) * batch_size] = features_batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
                break
        return features, labels

"""Establish convolutional base model for transfer learning"""
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
###############################################################################
###############################################################################
###############################################################################


"""Data Processing"""
total_data = pd.read_csv("leafClassification/train.csv").values
total_data = total_data[:,:2]
#np.random.shuffle(train_data1) #I skipped shufflinf since my data already shuffled
train_data1 = total_data[:round(len(total_data)*0.8),:]
total_labels = np.unique(train_data1[:,1])

###############################################################################
###############################################################################
###############################################################################
