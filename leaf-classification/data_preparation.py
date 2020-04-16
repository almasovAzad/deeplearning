@author: aza8223
"""




###############################################################################
"""Deep NN - Project1 - Azad Almasov"""
###############################################################################


"""
The OS module in Python provides a way of using operating system dependent
functionality. 
The functions that the OS module provides allows you to interface with the
underlying operating system that Python is running on – be that Windows, Mac or
Linux. 
You can find important information about your location or about the process.
"""
import os, shutil

import numpy as np
import pandas as pd
###############################################################################


original_dataset_dir = 'leafClassification/images'

base_dir = 'leaves'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

"""First I import datat classes names and picture labes so that i can create
appropriate folder for each of them"""
total_data = pd.read_csv("leafClassification/train.csv").values

total_data = total_data[:,:2]
#np.random.shuffle(total_data) #I prefer not to shuffle data since it has
                               #already shuffled.
train_data = total_data[:round(len(total_data)*0.6),:]
val_data = total_data[round(len(total_data)*0.6):round(len(total_data)*0.8),:]
test_data = total_data[round(len(total_data)*0.8):,:]


train_indices = train_data[:,0]
val_indices = val_data[:,0]
test_indices = test_data[:,0]


"""Creating labels for classes for each training and val dataset"""
total_labels = np.concatenate([train_data[:,1], val_data[:,1]])

"""Get rid of repeated labels so that to get total number of classes (99)
in alphabetical order"""
total_labels = np.unique(total_labels)

total_labelss=[]
"""Assign indice to each class"""
for i in enumerate(total_labels):
    total_labelss.append(i)

total_labelss = dict([(value, key) for (key, value) in dict(total_labelss).items()])



"""Create folder for each classes:"""
"""Copying test and train data images. Further I will get validation data from
training data"""

for k,v in total_labelss.items():
    a = os.path.join(train_dir,k)
    os.mkdir(a)
    for i, j in train_data:
        if j in k:
            fname =  '{}.jpg'.format(i)
            src = os.path.join(original_dataset_dir,fname)
            dst = os.path.join(a,fname)
            shutil.copyfile(src, dst)
    print('total training ', k, ' data: ', len(os.listdir(a)))

for k,v in total_labelss.items():
    a = os.path.join(validation_dir,k)
    os.mkdir(a)
    for i, j in val_data:
        if j in k:
            fname =  '{}.jpg'.format(i)
            src = os.path.join(original_dataset_dir,fname)
            dst = os.path.join(a,fname)
            shutil.copyfile(src, dst)
    print('total validation ', k, ' data: ', len(os.listdir(a)))
            
for k,v in total_labelss.items():
    a = os.path.join(test_dir,k)
    os.mkdir(a)
    for i, j in test_data:
        if j in k:
            fname =  '{}.jpg'.format(i)
            src = os.path.join(original_dataset_dir,fname)
            dst = os.path.join(a,fname)
            shutil.copyfile(src, dst)
    print('total test ', k, ' data: ', len(os.listdir(a)))



"""Assign those indices as label to our test, training and validation data
as well as check if any label missing in them"""

for i,j in enumerate(train_data[:,1]):
    if train_data[i,1] in list(total_labelss.keys()):
         train_data[i,1]=total_labelss[j]
    else:
         print(j, ' label is missing in training data')

for k,v in enumerate(val_data[:,1]):
    if val_data[k,1] in list(total_labelss.keys()):
         val_data[k,1]=total_labelss[v]
    else:
         print(v, ' label is missing in validation data')
   
for k,v in enumerate(test_data[:,1]):
    if test_data[k,1] in list(total_labelss.keys()):
         test_data[k,1]=total_labelss[v]
    else:
         print(v, ' label is missing in validation data')



"""Train, validation and test data labels:"""   
train_labels = train_data[:,1]
val_labels = val_data[:,1]
test_labels = test_data[:,1]

"""Vectorize labels"""
from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
val_labels = to_categorical(val_labels)
