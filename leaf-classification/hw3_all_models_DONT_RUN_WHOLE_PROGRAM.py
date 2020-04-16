# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:35:27 2018

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
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""Basic dense model 1"""
from keras import layers
from keras import models
from keras import regularizers



modelDense1 = models.Sequential()
modelDense1.add(layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
modelDense1.add(layers.MaxPooling2D((2,2)))
modelDense1.add(layers.MaxPooling2D((2,2)))
modelDense1.add(layers.Flatten())
modelDense1.add(layers.Dropout(0.5))
modelDense1.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
modelDense1.add(layers.Dropout(0.5))
modelDense1.add(layers.Dense(256,  kernel_regularizer=regularizers.l2(0.001), activation='relu'))
modelDense1.add(layers.Dropout(0.5))
modelDense1.add(layers.Dense(99, activation='softmax'))

modelDense1.summary()

"""Configuring the model for training"""
from keras import optimizers
modelDense1.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = len(val_data)

historyDense1 = modelDense1.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
modelDense1.save('dense_model1')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = historyDense1.history['acc']
val_acc = historyDense1.history['val_acc']
loss = historyDense1.history['loss']
val_loss = historyDense1.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################


"""BUILDING YOUR NETWORK-1"""
from keras import layers
from keras import models

model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Flatten())
model1.add(layers.Dense(512, activation='relu'))
model1.add(layers.Dense(99, activation='softmax'))

model1.summary()


"""Configuring the model for training"""
from keras import optimizers
model1.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = len(val_data)
print(validation_steps)

history1 = model1.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model1.save('simple_model1')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history1.history['acc']
val_acc = history1.history['val_acc']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK2"""
from keras import layers
from keras import models

model2 = models.Sequential()
model2.add(layers.Conv2D(64, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(256, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(512, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dense(99, activation='softmax'))

model2.summary()


"""Configuring the model for training"""
from keras import optimizers
model1.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = len(val_data)
print(validation_steps)

history2 = model2.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model2.save('simple_model2_double_hidden_units')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""Model3 Using Data Augumentation"""


"""I havent used "width shift", "height shift" and "shear range" since it may
cause feature coincidence of classes"""

datagen = ImageDataGenerator(
            rotation_range=40,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

"""Displaying some randomly augmented training images"""

from keras.preprocessing import image
fnames = [os.path.join(a, fname) for
          fname in os.listdir(a)]
img_path = fnames[0]
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()





"""Defining a new convnet that includes dropout"""

model3 = models.Sequential()
model3.add(layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(128, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(128, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Flatten())
model3.add(layers.Dropout(0.5))
model3.add(layers.Dense(512, activation='relu'))
model3.add(layers.Dense(99, activation='softmax'))
model3.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=1e-4),
                   metrics=['acc'])


"""Training the convnet using data-augmentation generators"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=32,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=32,
                    class_mode='categorical')

history3 = model3.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=50)


"""Saving the model"""
model3.save('augmentated_model1')

"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-4-with 1 more dense layers"""
from keras import layers
from keras import models

model4 = models.Sequential()
model4.add(layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(64, (3, 3), activation='relu'))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(128, (3, 3), activation='relu'))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(128, (3, 3), activation='relu'))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Flatten())
model4.add(layers.Dense(512, activation='relu'))
model4.add(layers.Dense(256, activation='relu'))
model4.add(layers.Dense(99, activation='softmax'))

model4.summary()


"""Configuring the model for training"""
from keras import optimizers
model4.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history4 = model4.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model4.save('simple_model4_with_more_dense_layers')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history4.history['acc']
val_acc = history4.history['val_acc']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-5-with 2 more dense layers"""
from keras import layers
from keras import models

model5 = models.Sequential()
model5.add(layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Conv2D(64, (3, 3), activation='relu'))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Conv2D(128, (3, 3), activation='relu'))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Conv2D(128, (3, 3), activation='relu'))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Flatten())
model5.add(layers.Dense(512, activation='relu'))
model5.add(layers.Dense(256, activation='relu'))
model5.add(layers.Dense(256, activation='relu'))
model5.add(layers.Dense(99, activation='softmax'))

model5.summary()


"""Configuring the model for training"""
from keras import optimizers
model5.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history5 = model5.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model5.save('simple_model4_with_2_more_dense_layers')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history5.history['acc']
val_acc = history5.history['val_acc']
loss = history5.history['loss']
val_loss = history5.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-6-with 2 more dense layers and regularization"""
from keras import layers
from keras import models
from keras import regularizers

model6 = models.Sequential()
model6.add(layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model6.add(layers.MaxPooling2D((2, 2)))
model6.add(layers.Conv2D(64, (3, 3), activation='relu'))
model6.add(layers.MaxPooling2D((2, 2)))
model6.add(layers.Conv2D(128, (3, 3), activation='relu'))
model6.add(layers.MaxPooling2D((2, 2)))
model6.add(layers.Conv2D(128, (3, 3), activation='relu'))
model6.add(layers.MaxPooling2D((2, 2)))
model6.add(layers.Flatten())
model6.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model6.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model6.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model6.add(layers.Dense(99, activation='softmax'))

model6.summary()


"""Configuring the model for training"""
from keras import optimizers
model6.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history6 = model6.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model6.save('simple_model4_with_2_more_dense_layers_and_regularization')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history6.history['acc']
val_acc = history6.history['val_acc']
loss = history6.history['loss']
val_loss = history6.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-7-with more conv layers and regularization"""

from keras import layers
from keras import models
from keras import regularizers

model7 = models.Sequential()
model7.add(layers.Conv2D(64, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model7.add(layers.Conv2D(64, (3, 3), activation='relu'))
model7.add(layers.MaxPooling2D((2, 2)))
model7.add(layers.Conv2D(128, (3, 3), activation='relu'))
model7.add(layers.Conv2D(128, (3, 3), activation='relu'))
model7.add(layers.MaxPooling2D((2, 2)))
model7.add(layers.Conv2D(256, (3, 3), activation='relu'))
model7.add(layers.Conv2D(256, (3, 3), activation='relu'))
model7.add(layers.Conv2D(256, (3, 3), activation='relu'))
model7.add(layers.MaxPooling2D((2, 2)))
model7.add(layers.MaxPooling2D((2, 2)))
model7.add(layers.Flatten())
model7.add(layers.Dense(512,  activation='relu'))
model7.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model7.add(layers.Dense(99, activation='softmax'))

model7.summary()


"""Configuring the model for training"""
from keras import optimizers
model7.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history7 = model7.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model7.save('simple_model4_with_2_more_conv_layers_and_regularization')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history7.history['acc']
val_acc = history7.history['val_acc']
loss = history7.history['loss']
val_loss = history7.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-8-with more conv layers and regularization and drop out"""

from keras import layers
from keras import models
from keras import regularizers

model8 = models.Sequential()
model8.add(layers.Conv2D(64, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model8.add(layers.Conv2D(64, (3, 3), activation='relu'))
model8.add(layers.MaxPooling2D((2, 2)))
model8.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8.add(layers.MaxPooling2D((2, 2)))
model8.add(layers.Conv2D(256, (3, 3), activation='relu'))
model8.add(layers.Conv2D(256, (3, 3), activation='relu'))
model8.add(layers.Conv2D(256, (3, 3), activation='relu'))
model8.add(layers.MaxPooling2D((2, 2)))
model8.add(layers.MaxPooling2D((2, 2)))
model8.add(layers.Flatten())
model8.add(layers.Dropout(0.5))
model8.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model8.add(layers.Dropout(0.5))
model8.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model8.add(layers.Dropout(0.5))
model8.add(layers.Dense(99, activation='softmax'))

model8.summary()


"""Configuring the model for training"""
from keras import optimizers
model8.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 10

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')




"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history8 = model8.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model8.save('simple_model8_64-64-mp-128-128-mp-256-256-256-mp-mp-flt-dp-1024-dp-512-dp')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history8.history['acc']
val_acc = history8.history['val_acc']
loss = history8.history['loss']
val_loss = history8.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


"""Prediction"""

[losss, accur]  = model8.evaluate_generator(test_generator, steps=len(test_generator))
print("Accuracy of the model is  ", accur)

###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-8aa-with more conv layers and regularization and drop out
WITH AUGMENTATION"""
datagen = ImageDataGenerator(
            rotation_range=40,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

"""Displaying some randomly augmented training images"""

from keras.preprocessing import image
fnames = [os.path.join(a, fname) for
          fname in os.listdir(a)]
img_path = fnames[0]
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()


from keras import layers
from keras import models
from keras import regularizers


model8aa = models.Sequential()
model8aa.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model8aa.add(layers.Conv2D(64, (3, 3), activation='relu'))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8aa.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.Flatten())
model8aa.add(layers.Dropout(0.5))
model8aa.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model8aa.add(layers.Dropout(0.5))
model8aa.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model8aa.add(layers.Dropout(0.5))
model8aa.add(layers.Dense(99, activation='softmax'))

model8aa.summary()


"""Configuring the model for training"""
from keras import optimizers
model8aa.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

"""Training the convnet using data-augmentation generators"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=15,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=15,
                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(150, 150),
                    batch_size=15,
                    class_mode='categorical')

history8aa = model8aa.fit_generator(
            train_generator,
            steps_per_epoch=64,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=22)



"""Saving the model"""
model8aa.save('simple_model8aa_with_2_more_conv_layers(with_double_maxpooling)_and_regularization_and_drop_out')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history8aa.history['acc']
val_acc = history8aa.history['val_acc']
loss = history8aa.history['loss']
val_loss = history8aa.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

"""Prediction"""
[losss, accur]  = model8aa.evaluate_generator(test_generator, steps=len(test_generator))
print("Accuracy of the model is  ", accur)


###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-8B-with more conv layers and regularization and drop out-B"""

from keras import layers
from keras import models
from keras import regularizers

model8b = models.Sequential()
model8b.add(layers.Conv2D(64, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model8b.add(layers.Conv2D(64, (3, 3), activation='relu'))
model8b.add(layers.MaxPooling2D((2, 2)))
model8b.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8b.add(layers.MaxPooling2D((2, 2)))
model8b.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8b.add(layers.MaxPooling2D((2, 2)))
model8b.add(layers.Conv2D(256, (3, 3), activation='relu'))
model8b.add(layers.Conv2D(256, (3, 3), activation='relu'))
model8b.add(layers.MaxPooling2D((2, 2)))
model8b.add(layers.Conv2D(256, (3, 3), activation='relu'))
model8b.add(layers.Flatten())
model8b.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model8b.add(layers.Dropout(0.5))
model8b.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model8b.add(layers.Dense(99, activation='softmax'))

model8b.summary()


"""Configuring the model for training"""
from keras import optimizers
model8b.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history8b = model8b.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model8b.save('simple_model4B_with_2_more_conv_layers(with_double_maxpooling)_and_regularization_and_drop_out')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history8b.history['acc']
val_acc = history8b.history['val_acc']
loss = history8b.history['loss']
val_loss = history8b.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-9-with more conv layers and regularization and drop out
and DATA AUGMENTATION"""

from keras import layers
from keras import models
from keras import regularizers

"""I havent used "width shift", "height shift" and "shear range" since it may
cause feature coincidence of classes"""

datagen = ImageDataGenerator(
            rotation_range=40,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

"""Displaying some randomly augmented training images"""

from keras.preprocessing import image

fnames = [os.path.join(a, fname) for
          fname in os.listdir(a)]
img_path = fnames[0]
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()



model9 = models.Sequential()
model9.add(layers.Conv2D(64, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model9.add(layers.MaxPooling2D((2, 2)))
model9.add(layers.Conv2D(128, (3, 3), activation='relu'))
model9.add(layers.Conv2D(128, (3, 3), activation='relu'))
model9.add(layers.MaxPooling2D((2, 2)))
model9.add(layers.Conv2D(256, (3, 3), activation='relu'))
model9.add(layers.Conv2D(256, (3, 3), activation='relu'))
model9.add(layers.Conv2D(256, (3, 3), activation='relu'))
model9.add(layers.MaxPooling2D((2, 2)))
model9.add(layers.Flatten())
model9.add(layers.Dropout(0.5))
model9.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
#model9.add(layers.Dropout(0.5))
model9.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model9.add(layers.Dense(99, activation='softmax'))

model9.summary()


"""Configuring the model for training"""
from keras import optimizers
model9.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Training the convnet using data-augmentation generators"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=32,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=32,
                    class_mode='categorical')

history9 = model9.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=50)



"""Saving the model"""
model9.save('simple_model4_with_2_more_dense_layers_and_regularization_and_drop_out_withDATA_Augmentation')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history9.history['acc']
val_acc = history9.history['val_acc']
loss = history9.history['loss']
val_loss = history9.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################
"""BUILDING YOUR NETWORK-10(64-128-256-512-dropout-2048-1024-regularizer"""
from keras import layers
from keras import models

model10 = models.Sequential()
model10.add(layers.Conv2D(64, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model10.add(layers.MaxPooling2D((2, 2)))
model10.add(layers.Conv2D(128, (3, 3), activation='relu'))
model10.add(layers.MaxPooling2D((2, 2)))
model10.add(layers.Conv2D(256, (3, 3), activation='relu'))
model10.add(layers.MaxPooling2D((2, 2)))
model10.add(layers.Conv2D(512, (3, 3), activation='relu'))
model10.add(layers.MaxPooling2D((2, 2)))
model10.add(layers.Flatten())
model10.add(layers.Dropout(0.4))
model10.add(layers.Dense(2048,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model10.add(layers.Dropout(0.4))
model10.add(layers.Dense(2048, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model10.add(layers.Dense(99, activation='softmax'))
model10.summary()


"""Configuring the model for training"""
from keras import optimizers
model10.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 15

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history10 = model10.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model10.save('simple_model2_2048-1024_dense')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history10.history['acc']
val_acc = history10.history['val_acc']
loss = history10.history['loss']
val_loss = history10.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()






###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################


"""BUILDING YOUR NETWORK-simple_model11_64_mp_mp_128_128_mp_mp_256_256_flt_dp_1024_dp_512_dp"""

from keras import layers
from keras import models
from keras import regularizers

model11 = models.Sequential()
model11.add(layers.Conv2D(64, (3, 3), activation='relu',
                         input_shape=(150, 150, 3)))
model11.add(layers.MaxPooling2D((2, 2)))
model11.add(layers.MaxPooling2D((2, 2)))
model11.add(layers.Conv2D(128, (3, 3), activation='relu'))
model11.add(layers.Conv2D(128, (3, 3), activation='relu'))
model11.add(layers.MaxPooling2D((2, 2)))
model11.add(layers.MaxPooling2D((2, 2)))
model11.add(layers.Conv2D(256, (3, 3), activation='relu'))
model11.add(layers.Conv2D(256, (3, 3), activation='relu'))
model11.add(layers.Flatten())
model11.add(layers.Dropout(0.5))
model11.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model11.add(layers.Dropout(0.5))
model11.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model11.add(layers.Dropout(0.5))
model11.add(layers.Dense(99, activation='softmax'))

model11.summary()


"""Configuring the model for training"""
from keras import optimizers
model11.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 10

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')




"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history11 = model11.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model11.save('simple_model11_64_mp_mp_128_128_mp_mp_256_256_flt_dp_1024_dp_512_dp')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history11.history['acc']
val_acc = history11.history['val_acc']
loss = history11.history['loss']
val_loss = history11.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


"""Prediction"""

[losss, accur]  = model11.evaluate_generator(test_generator, steps=len(test_data))
print("Accuracy of the model is  ", accur)

###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################


"""BUILDING YOUR NETWORK-simple_model12_64_64_mp_mp_128_128_mp_mp_mp_flt_dp_256_dp"""

from keras import layers
from keras import models
from keras import regularizers

model12 = models.Sequential()
model12.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model12.add(layers.Conv2D(64, (3, 3), activation='relu'))
model12.add(layers.MaxPooling2D((2, 2)))
model12.add(layers.MaxPooling2D((2, 2)))
model12.add(layers.Conv2D(128, (3, 3), activation='relu'))
model12.add(layers.Conv2D(128, (3, 3), activation='relu'))
model12.add(layers.MaxPooling2D((2, 2)))
model12.add(layers.MaxPooling2D((2, 2)))
model12.add(layers.MaxPooling2D((2, 2)))
model12.add(layers.Flatten())
model12.add(layers.Dropout(0.5))
model12.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model12.add(layers.Dropout(0.5))
model12.add(layers.Dense(99, activation='softmax'))

model12.summary()


"""Configuring the model for training"""
from keras import optimizers
model12.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 6

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')




"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history12 = model12.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=175,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model12.save('simple_model12_64_64_mp_mp_128_128_mp_mp_mp_flt_dp_256_dp')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history12.history['acc']
val_acc = history12.history['val_acc']
loss = history12.history['loss']
val_loss = history12.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


"""Prediction"""

[losss, accur]  = model12.evaluate_generator(test_generator, steps=len(test_generator))
print("Accuracy of the model is  ", accur)

###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################


"""BUILDING YOUR NETWORK-simple_model13_64_64_64_mp_mp_64_64_mp_mp_mp_flt_dp_256_dp"""

from keras import layers
from keras import models
from keras import regularizers

model13 = models.Sequential()
model13.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model13.add(layers.Conv2D(64, (3, 3), activation='relu'))
model13.add(layers.Conv2D(64, (3, 3), activation='relu'))
model13.add(layers.MaxPooling2D((2, 2)))
model13.add(layers.MaxPooling2D((2, 2)))
model13.add(layers.Conv2D(128, (3, 3), activation='relu'))
model13.add(layers.MaxPooling2D((2, 2)))
model13.add(layers.MaxPooling2D((2, 2)))
model13.add(layers.MaxPooling2D((2, 2)))
model13.add(layers.Flatten())
model13.add(layers.Dropout(0.5))
model13.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model13.add(layers.Dropout(0.5))
model13.add(layers.Dense(99, activation='softmax'))

model13.summary()


"""Configuring the model for training"""
from keras import optimizers
model13.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


"""Data preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

batch_size = 6

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')




"""Lets look at output of one of these generators"""
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""Fitting the model using a batch generator"""
"""Before that you have to determine steps per epochs size yourself"""

steps_per_epoch = round(len(train_data)/batch_size)
print(steps_per_epoch)
validation_steps = round(len(val_data)/batch_size)
print(validation_steps)

history13 = model13.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=75,
            validation_data=validation_generator,
            validation_steps=validation_steps)


"""Saving the model"""
model13.save('simple_model14_64_64_64_mp_mp_128_mp_mp_mp_flt_dp_256_dp')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history13.history['acc']
val_acc = history13.history['val_acc']
loss = history13.history['loss']
val_loss = history13.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


"""Prediction"""

[losss, accur]  = model13.evaluate_generator(test_generator, steps=len(test_generator))
print("Accuracy of the model is  ", accur)
###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################

"""BUILDING YOUR NETWORK-8aa-with more conv layers and regularization and drop out
WITH AUGMENTATION"""
datagen = ImageDataGenerator(
            rotation_range=40,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

"""Displaying some randomly augmented training images"""

from keras.preprocessing import image
fnames = [os.path.join(a, fname) for
          fname in os.listdir(a)]
img_path = fnames[0]
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()


from keras import layers
from keras import models
from keras import regularizers


model8aa = models.Sequential()
model8aa.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model8aa.add(layers.Conv2D(64, (3, 3), activation='relu'))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8aa.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.MaxPooling2D((2, 2)))
model8aa.add(layers.Flatten())
model8aa.add(layers.Dropout(0.5))
model8aa.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model8aa.add(layers.Dropout(0.5))
model8aa.add(layers.Dense(99, activation='softmax'))

model8aa.summary()

"""Configuring the model for training"""
from keras import optimizers
model8aa.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

"""Training the convnet using data-augmentation generators"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=10,
                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=10,
                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='categorical')

history8aa = model8aa.fit_generator(
            train_generator,
            steps_per_epoch=99,
            epochs=300,
            validation_data=validation_generator,
            validation_steps=round(99*len(val_data)/len(train_data)))



"""Saving the model"""
model8aa.save('simple_model12_64_64_mp_mp_128_128_mp_mp_mp_flt_dp_256_Augmented')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = history8aa.history['acc']
val_acc = history8aa.history['val_acc']
loss = history8aa.history['loss']
val_loss = history8aa.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

"""Prediction"""
[losss, accur]  = model8aa.evaluate_generator(test_generator, steps=len(test_generator))
print("Accuracy of the model is  ", accur)


###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################

""" FINE TUNING WITH DENSE NETWORK"""

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

from keras import models
from keras import layers
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
"""Feature Extraction"""


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


train_features, train_labels = extract_features(train_dir, len(train_data))
validation_features, validation_labels = extract_features(validation_dir, len(val_data))
test_features, test_labels = extract_features(test_dir, len(test_data))


train_features = np.reshape(train_features, (len(train_data), 4 * 4 * 512))
validation_features = np.reshape(validation_features, (len(val_data), 4 * 4 * 512))
test_features = np.reshape(test_features, (len(test_data), 4 * 4 * 512))





modelPT = models.Sequential()
modelPT.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
modelPT.add(layers.Dropout(0.5))
modelPT.add(layers.Dense(99, activation='softmax'))

modelPT.summary()

modelPT.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

historyPT = modelPT.fit(train_features, train_labels,
                    epochs=200,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))

"""Saving the model"""
modelPT.save('modelPT_256_dense')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = historyPT.history['acc']
val_acc = historyPT.history['val_acc']
loss = historyPT.history['loss']
val_loss = historyPT.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


"""Prediction"""
[losss, accur]  = modelPT.evaluate(test_features, test_labels)
print("Accuracy of the model is  ", accur)


###############################################################################
                       #         #
                        #       #
                         #     #
                          #   #
                           # #
                            #
###############################################################################

""" FINE TUNING WITH DENSE NETWORK with LAST CONVOLUTIONAL NETWORK"""

from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))




"""Feature Extraction"""
from keras import models
from keras import layers



"""Unfreeze last layer"""
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


modelPTC = models.Sequential()
modelPTC.add(conv_base)
modelPTC.add(layers.Flatten())
modelPTC.add(layers.Dense(256, activation='relu'))
modelPTC.add(layers.Dense(99, activation='softmax'))

modelPTC.summary()
        
"""Generate Data"""
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(150, 150),
                batch_size=10,
                class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                validation_dir,
                target_size=(150, 150),
                batch_size=10,
                class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(150, 150),
                batch_size=10,
                class_mode='categorical')


modelPTC.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

historyPTC = modelPTC.fit_generator(
        train_generator,
        steps_per_epoch=99,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=round(99*len(val_data)/len(train_data)))

"""Saving the model"""
modelPTC.save('modelPTC_256_dense')


"""Displaying curves of loss and accuracy during training"""
import matplotlib.pyplot as plt
acc = historyPTC.history['acc']
val_acc = historyPTC.history['val_acc']
loss = historyPTC.history['loss']
val_loss = historyPTC.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


"""Prediction"""
[losss, accur]  = modelPTC.evaluate_generator(test_generator, steps=len(test_generator))
print("Accuracy of the model is  ", accur)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
                            
"""Since I have used transfer learning, then my convolutional part is VG16,
therefore, Filter visualization will give exactly the same images as in the
book. Therefore, here, to be different, I will show results of my simpe model,
which gave 75 percent accuracy,-model 6 for visualization of filters and 
activations."""

"""VISUALIZING INTERMEDIATE ACTIVATIONS"""

from keras.models import load_model
model = load_model('simple_model6')
model.summary()

img_test_ex_dir = 'leaves/test/Acer_Capillipes'
ex = os.listdir(img_test_ex_dir)

img_path = os.path.join(img_test_ex_dir,ex[0])


from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)


import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')

third_layer_activation = activations[2]
plt.matshow(third_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(third_layer_activation[0, :, :, 7], cmap='viridis')
plt.matshow(third_layer_activation[0, :, :, 30], cmap='viridis')


"""Visualizing every channel in every intermediate activation"""

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
            scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
"""VISUALISING CONVNET FILTERS"""

from keras.applications import VGG16
from keras import backend as K
#model = VGG16(weights='imagenet',
#              include_top=False)

from keras.models import load_model
model = load_model('simple_model6')
model.summary()

layer_name = 'conv2d_90'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, model.input)[0]
iterate = K.function([model.input], [loss, grads])
import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

step = 1.
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    
    input_img_data += grads_value * step
    

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
#    x = np.clip(x, 0, 1)
#    x *= 255
#    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    
    return deprocess_image(img)

plt.imshow(generate_pattern('conv2d_90', 0))

"""Generating a grid of all filter response patterns in a layer"""
layer_name = 'conv2d_90'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img
                
plt.figure(figsize=(20, 20))
plt.imshow(results)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

"""VISUALISING HEATMAPS OF CLASS ACTIVATION"""
"""However, to visualize heatmap I used my transfer learning model which is
my main model"""
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')                        

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
                            
acer_capilipes_output = model.output[:, 386]           
last_conv_layer = model.get_layer('block5_conv3')
  
grads = K.gradients(acer_capilipes_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
                            
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
                            
import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('leaves/test/Acer_Capillipes/asd.jpg', superimposed_img) 


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

"""Doing Iterative K-Fold Validation"""


from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

from keras import models
from keras import layers
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os, shutil

import numpy as np
import pandas as pd
###############################################################################





total_data = pd.read_csv("leafClassification/train.csv").values
total_data = total_data[:,:2]

train_data1 = total_data[:round(len(total_data)*0.8),:]


total_labels = np.unique(train_data1[:,1])

k = 4 #I took k as 3 since my data is already very small.
num_val_samples = len(train_data1) // k
num_epochs = 100
all_scores = []
all_val_acc = []
all_train_acc = []



   





#np.random.shuffle(train_data1)



         

for bb in range(k):
    
    
    
    
    print('processing fold #', bb)

    total_data = pd.read_csv("leafClassification/train.csv").values
    total_data = total_data[:,:2]

    train_data1 = total_data[:round(len(total_data)*0.8),:]
    
    original_dataset_dir = 'leafClassification/images'

    base_dir = 'leaves'
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir,'train')
    validation_dir = os.path.join(base_dir,'validation')
    test_dir = os.path.join(base_dir,'test')
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
          
          
    
    
    """Train, validation and test data labels:"""   

    
    val_data = train_data1[bb * num_val_samples: (bb + 1) * num_val_samples,:]
    train_data = np.concatenate(
                  [train_data1[:bb * num_val_samples,:],
                   train_data1[(bb + 1) * num_val_samples:,:]])
 
    
    
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
   
    train_labels = train_data[:,1]
    val_labels = val_data[:,1]

    """Vectorize labels"""
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)





    
    


    """TRAINING PROCESS"""
    
    

    datagen = ImageDataGenerator(rescale=1./255)
    """Feature Extraction"""


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


    train_features, train_labels = extract_features(train_dir, len(train_data))
    validation_features, validation_labels = extract_features(validation_dir, len(val_data))


    train_features = np.reshape(train_features, (len(train_data), 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (len(val_data), 4 * 4 * 512))





    modelPT = models.Sequential()
    modelPT.add(layers.Dense(512, activation='relu', input_dim=4 * 4 * 512))
    modelPT.add(layers.Dropout(0.5))
    modelPT.add(layers.Dense(99, activation='softmax'))

    modelPT.summary()

    modelPT.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

    historyPT = modelPT.fit(train_features, train_labels,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels),
                    verbose=0)
    

    validation_acc = historyPT.history['val_acc']
    all_val_acc.append(validation_acc)      

    training_acc = historyPT.history['acc']
    all_train_acc.append(training_acc) 
    
    shutil.rmtree('leaves')
          
    del train_data
    del val_data
    del train_labels
    del val_labels
    del train_features
    del validation_features
    del total_data
    del train_data1
 



average_val_acc = [np.mean([x[i] for x in all_val_acc]) for i in range(num_epochs)]
average_train_acc = [np.mean([x[i] for x in all_train_acc]) for i in range(num_epochs)]

"""Plotting average val_acc and train_acc"""
import matplotlib.pyplot as plt

epochs = range(1, len(average_val_acc) + 1)
plt.plot(epochs, average_train_acc, 'bo', label='Training acc')
plt.plot(epochs, average_val_acc, 'b', label='Validation acc')
plt.title('k-foldTraining and validation accuracy')
plt.legend()
plt.figure()
plt.show()




"""TRAINING whole TRAINING DATA"""
    

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
#np.random.shuffle(total_data)
train_data = total_data[:round(len(total_data)*0.8),:]
test_data = total_data[round(len(total_data)*0.8):,:]


train_indices = train_data[:,0]
test_indices = test_data[:,0]




"""Get rid of repeated labels so that to get total number of classes (99)
in alphabetical order"""
total_labels = np.unique(train_data[:,1])


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

   
for k,v in enumerate(test_data[:,1]):
    if test_data[k,1] in list(total_labelss.keys()):
         test_data[k,1]=total_labelss[v]
    else:
         print(v, ' label is missing in validation data')



"""Train, validation and test data labels:"""   
train_labels = train_data[:,1]
test_labels = test_data[:,1]

"""Vectorize labels"""
from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
    

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

from keras import models
from keras import layers
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
"""Feature Extraction"""


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


train_features, train_labels = extract_features(train_dir, len(train_data))
test_features, test_labels = extract_features(test_dir, len(test_data))


train_features = np.reshape(train_features, (len(train_data), 4 * 4 * 512))
test_features = np.reshape(test_features, (len(test_data), 4 * 4 * 512))


modelPT = models.Sequential()
modelPT.add(layers.Dense(512, activation='relu', input_dim=4 * 4 * 512))
modelPT.add(layers.Dropout(0.5))
modelPT.add(layers.Dense(99, activation='softmax'))

modelPT.summary()

modelPT.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

historyPT = modelPT.fit(train_features, train_labels,
                    epochs=100,
                    batch_size=batch_size)

"""Saving the model"""
modelPT.save('modelPT_WHOLE_512_dense')





"""Prediction"""
[losss, accur]  = modelPT.evaluate(test_features, test_labels)
print("Accuracy of the model is  ", accur)
          
          
          

    