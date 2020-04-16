"""3b: Combining CNNs and RNNs to process long sequences"""
""" not bad and it is fast"""


if input_size >5:
    model = Sequential()
    model.add(layers.Conv1D(32, input_size-5, activation='relu',
                        input_shape=(None, training_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(output_size, activation='tanh'))

    """COMPILE YOUR MODEL"""
    model.compile(optimizer=RMSprop(), loss='mae')
    

    """TRAINING YOUR MODEL"""
    epoch_size = 22
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
    #model.save('ConvRNN_combined_1day')
else:
    print("for 1 day sequence you cannot use Conv layer")
    
