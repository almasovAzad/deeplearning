"""1: Training and evaluating a densely connected model"""
"""I have tried different kind of architectures hidden units etc, but found this
useful since it does not overfit and I got lower loss - 0.1015 (unnormilized)"""
from keras.models import Sequential
model = Sequential()
model.add(layers.Flatten(input_shape=(input_size, training_data.shape[-1])))
model.add(layers.Dense(64,activation='tanh'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32,activation='tanh'))
model.add(layers.Dense(output_size,activation='tanh'))


"""COMPILE YOUR MODEL"""
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='mae')


"""TRAINING YOUR MODEL"""
epoch_size = 20
batch_size = 32
history = model.fit(training_data,
                    training_labels,
                    epochs=epoch_size,
                    batch_size=batch_size,
                    validation_data = (val_data, val_labels))


"""Plotting results"""
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation losses')
plt.legend()
plt.show()


"""PREDICTION - TESTING DATA"""
test_loss = model.evaluate(test_data, test_labels)
print('normalized test_loss:', test_loss)
print('unnormalized test_loss:', test_loss*std[0])
 
"""Save your model:"""
#model.save('dense_model_1day')
