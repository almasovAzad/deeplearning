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
