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
