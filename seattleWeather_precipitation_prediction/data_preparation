@author: aza8223
"""

"""Project2"""


###############################################################################

"""Importing important libraries"""

import numpy as np
import pandas as pd
from keras import layers
from keras import optimizers
import math
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop


###############################################################################


"""Preparing data"""
data =[]
if data:
    del data

total_data = pd.read_csv("seattleWeather_1948-2017.csv").values
data = total_data[:,1:]

"""Check if there is nan (missing) data and replace them with their next data:"""
"""Here i have used while loop for the case when oreceding samples all nan replacement
keeps going until get reasonable neighbor value"""
data = pd.DataFrame(data=data)
while 1:
    for j, kays in enumerate(data.loc[0,:]):
        for i, kay in enumerate(data.loc[:,0]):
            if math.isnan(data.loc[i,j]):
                data.loc[i,j]=data.loc[i+1,j]
                print("sample ", i, "feature", j, " was missing and replaced by its next samnple")
    if not data.isnull().any().any():
        break
data = np.asarray(data).astype('float32')
        
"""Change true and false to 1 and 0"""
for j, rain in enumerate(data[:,3]):
    if data[j, 3]==True:
        data[j,3]=1
    else:
        data[j,3]=0

data = data[:,:3] #If it rains or not is not important feature for the determination
#of amount of rain.
data = np.asarray(data).astype('float32')

"""Creating descriptive and target features"""
num_data = len(data)
output_size = 7 #Days to be predicted. They are fixed
input_size = 14 #Sequence of days to be descriptive feature. You can modify it
# as given in the problem: 1 day, 7 days, 14 days, 1 months.


"""Create data descriptime sequential features with the shape of sample*times*features"""
data_feat = np.zeros((num_data-(output_size+input_size),input_size,len(data[0])))
data_label = np.zeros((num_data-(output_size+input_size),output_size))
for i in range(num_data - (output_size+input_size)):
    data_feat[i] = data[i:i+input_size]
    data_label[i] = data[i+input_size:i+input_size+output_size,0]

"""Seperating data into dry and wet days"""
"""
To do so, i calculated mean of each output (7days that to be predicted)
then i compared that output with mean of all labels, and thus i devided my data
for dry week and wet week
"""
mean_each_output = data_label[:,:].mean(axis=1)
mean_all_data = np.nanmean(mean_each_output)

positive_data = []
positive_label = []
negative_data = []
negative_label = []

for i in range(len(data_label)):
    if mean_each_output[i]<=mean_all_data:
        negative_data.append(data_feat[i]) 
        negative_label.append(data_label[i])
    else:
        positive_data.append(data_feat[i]) 
        positive_label.append(data_label[i])

positive_data = np.asarray(positive_data).astype('float32')
positive_data_part1 = positive_data[:round(len(positive_data)/3)]
positive_data_part2 = positive_data[round(len(positive_data)/3):round(2*len(positive_data)/3)]
positive_data_part3 = positive_data[round(2*len(positive_data)/3):]

positive_label = np.asarray(positive_label).astype('float32')
positive_label_part1 = positive_label[:round(len(positive_data)/3)]
positive_label_part2 = positive_label[round(len(positive_data)/3):round(2*len(positive_data)/3)]
positive_label_part3 = positive_label[round(2*len(positive_data)/3):]

negative_data = np.asarray(negative_data).astype('float32')
negative_data_part1 = negative_data[:round(len(negative_data)/3)]
negative_data_part2 = negative_data[round(len(negative_data)/3):round(2*len(negative_data)/3)]
negative_data_part3 = negative_data[round(2*len(negative_data)/3):]

negative_label = np.asarray(negative_label).astype('float32')
negative_label_part1 = negative_label[:round(len(negative_data)/3)]
negative_label_part2 = negative_label[round(len(negative_data)/3):round(2*len(negative_data)/3)]
negative_label_part3 = negative_label[round(2*len(negative_data)/3):]

"""Create training, test, validation data and labels using 1/3 partion of both
negative and positive sets:"""

import itertools
training_data = []
for item in itertools.chain(positive_data_part1,negative_data_part1):
    training_data.append(item)

training_labels = []
for item in itertools.chain(positive_label_part1,negative_label_part1):
    training_labels.append(item)
    
test_data = []
for item in itertools.chain(positive_data_part2,negative_data_part2):
    test_data.append(item)
    
test_labels = []
for item in itertools.chain(positive_label_part2,negative_label_part2):
    test_labels.append(item)

val_data = []
for item in itertools.chain(positive_data_part3,negative_data_part3):
    val_data.append(item)
    
val_labels = []
for item in itertools.chain(positive_label_part3,negative_label_part3):
    val_labels.append(item)


training_data = np.asarray(training_data).astype('float32')
training_labels = np.asarray(training_labels).astype('float32')

test_data = np.asarray(test_data).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

val_data = np.asarray(val_data).astype('float32')
val_labels = np.asarray(val_labels).astype('float32')

"""Shuffle data and labels:"""

from random import shuffle

ind_list = [i for i in range(len(training_data))]
shuffle(ind_list)
training_data  = training_data[ind_list, :, :]
training_labels = training_labels[ind_list, :]

ind_list = [i for i in range(len(val_data))]
shuffle(ind_list)
val_data  = val_data[ind_list, :, :]
val_labels = val_labels[ind_list, :]

ind_list = [i for i in range(len(test_data))]
shuffle(ind_list)
test_data  = test_data[ind_list, :, :]
test_labels = test_labels[ind_list, :]

#Normalize your all data based on mean std of your training data and training labels:
mean = training_data[:,:,:].mean(axis=0)
training_data[:,:,:] -= mean
std = np.std(training_data[:,:,:],axis=0)
training_data[:,:,:] /= std

val_data[:,:,:] -= mean
val_data[:,:,:] /= std

test_data[:,:,:] -= mean
test_data[:,:,:] /= std

mean = training_labels[:,:].mean(axis=0)
training_labels[:,:] -= mean
std = np.std(training_labels[:,:],axis=0)
training_labels[:,:] /= std

val_labels[:,:] -= mean
val_labels[:,:] /= std

test_labels[:,:] -= mean
test_labels[:,:] /= std

