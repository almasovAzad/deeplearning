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
