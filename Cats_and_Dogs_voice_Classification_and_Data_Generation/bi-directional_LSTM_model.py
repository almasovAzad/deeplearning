"""2c: Bidirectional RNN""" """32"""

model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(256, activation='tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

#
#model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
#              loss='binary_crossentropy',
#              metrics=['acc'])

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              metrics=['acc'], optimizer=sgd)

"""TRAINING YOUR MODEL"""
epoch_size = 20
batch_size = 16
history = model.fit(part_train_data_seq,
                    part_train_label,
                    epochs=epoch_size,
                    batch_size=batch_size,
                    validation_data=(val_data_seq,val_label))


"""Plotting results"""
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()   
    
"""PREDICTION - TESTING DATA"""
test_acc = model.evaluate(test_data_seq, test_label)[1]
print('test_acc:', test_acc)

#"""Save your model:"""
#model.save('bidirectional_dense320dp0.5_dense32_SGDoptim_REDIFINED')
