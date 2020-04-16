
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
