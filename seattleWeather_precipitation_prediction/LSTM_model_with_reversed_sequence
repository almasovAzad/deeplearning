"""2d: Training and evaluating an LSTM using reversed sequences  10.23"""

"""First reverse days (sequentions or times) in your training and validation data,
but not labels"""
"""tanh seems better choice even for hidden layers"""
x_train = [x[::-1] for x in training_data] #It will reverse days (times)
x_test = [x[::-1] for x in test_data]
x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')

x_val = [x[::-1] for x in val_data] #It will reverse days (times)
x_val = np.asarray(x_val).astype('float32')


model = Sequential()
model.add(layers.LSTM(32))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(output_size, activation='tanh'))

"""COMPILE YOUR MODEL"""
model.compile(optimizer=RMSprop(), loss='mae')


"""TRAINING YOUR MODEL"""
epoch_size = 20
batch_size = 32
history = model.fit(x_train,
                    training_labels,
                    epochs=epoch_size,
                    batch_size=batch_size,
                    validation_data = (x_val, val_labels))


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
#model.save('Reversed_LSTM_1day')
