"""4: Using DAG network"""
"""When I used different layer types I put here the best architecture and
diagram for my prediction"""
"""One input but Multiple output. Diagram is shown in the report"""

from keras import layers
from keras import Input
from keras.models import Model



"""Input layer:"""
inputt = Input(shape=(input_size,training_data.shape[-1]), dtype='float32', name='previous_days')
a = layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, activation='relu')(inputt)
a = layers.Dropout(0.4)(a)

"""Output layers for each day:"""
x = layers.Dense(32, activation='relu')(a)
x = layers.Dropout(0.4)(x)
day_1 = layers.Dense(1,activation='tanh', name='day1')(x)

y = layers.Dense(32, activation='relu')(a)
y = layers.Dropout(0.4)(y)
day_2 = layers.Dense(1,activation='tanh', name='day2')(y)

z = layers.Dense(32, activation='relu')(a)
z = layers.Dropout(0.4)(z)
day_3 = layers.Dense(1,activation='tanh', name='day3')(z)

v = layers.Dense(32, activation='relu')(a)
v = layers.Dropout(0.4)(v)
day_4 = layers.Dense(1,activation='tanh', name='day4')(v)

w = layers.Dense(32, activation='relu')(a)
w = layers.Dropout(0.4)(w)
day_5 = layers.Dense(1,activation='tanh', name='day5')(w)

b = layers.Dense(32, activation='relu')(a)
b = layers.Dropout(0.4)(b)
day_6 = layers.Dense(1,activation='tanh', name='day6')(b)

c = layers.Dense(32, activation='relu')(a)
c = layers.Dropout(0.4)(c)
day_7 = layers.Dense(1,activation='tanh', name='day7')(c)


"""Fully connected DAG model:"""
model = Model(inputt, [day_1, day_2, day_3, day_4, day_5, day_6, day_7])

"""Compiling:"""
"""I could add multiple losses but my problem isa regression so only loss here is mae"""
"""I can also define different loss weights for different outputs, but that would be
good to use it when we have different type of loss functions. Just in case I have
used different weights but it didnt affaect my results much"""

model.compile(optimizer=RMSprop(), loss='mae')


"""TRAINING YOUR MODEL. Here I will assign target labels for each days seperately"""
epoch_size = 20
batch_size = 32
history = model.fit(training_data,
                   [training_labels[:,0],
                    training_labels[:,1],
                    training_labels[:,2],
                    training_labels[:,3],
                    training_labels[:,4],
                    training_labels[:,5],
                    training_labels[:,6]],
                    epochs=epoch_size,
                    batch_size=batch_size,
                    validation_data = (val_data, 
                   [val_labels[:,0],
                    val_labels[:,1],
                    val_labels[:,2],
                    val_labels[:,3],
                    val_labels[:,4],
                    val_labels[:,5],
                    val_labels[:,6]]))



"""Plot losses for each day in different plots"""
"""Predict losses for each day seperately:"""

"""Day1:"""
loss = history.history['day1_loss']
val_loss = history.history['val_day1_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation losses for Day1')
plt.legend()
plt.show()


###################################################
"""Day2:"""
loss = history.history['day2_loss']
val_loss = history.history['val_day2_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation losses for Day2')
plt.legend()
plt.show()


###################################################
"""Day3:"""
loss = history.history['day3_loss']
val_loss = history.history['val_day3_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation losses for Day3')
plt.legend()
plt.show()



###################################################
"""Day4:"""
loss = history.history['day4_loss']
val_loss = history.history['val_day4_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation losses for Day4')
plt.legend()
plt.show()


###################################################
"""Day5:"""
loss = history.history['day5_loss']
val_loss = history.history['val_day5_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation losses for Day5')
plt.legend()
plt.show()


###################################################
"""Day6:"""
loss = history.history['day6_loss']
val_loss = history.history['val_day6_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation losses for Day6')
plt.legend()
plt.show()


###################################################
"""Day7:"""
loss = history.history['day7_loss']
val_loss = history.history['val_day7_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation losses for Day7')
plt.legend()
plt.show()


###################################################
###################################################
"""PREDICTION - TESTING DATA for each days both normalized and unnormalized
for DAG model"""
test_LossAndAcc = model.evaluate(test_data, [i for i in np.transpose(test_labels)])
test_losses = test_LossAndAcc[1:]
for i, k in enumerate(test_losses):
    print('normalized test_loss of ', 'day', i+1, 'is', test_losses[i])
    print('unnormalized test_loss of ', 'day', i+1, 'is', test_losses[i]*std[0])


###################################################
###################################################
"""Base case for each day and mean of mae:"""
preds = np.mean(val_data[:, :, 0], axis=1)
day = np.zeros((val_labels.shape[1], val_labels.shape[0]))
mae_base1 = np.zeros((val_labels.shape[1],))
for i,j in enumerate(np.transpose(val_labels)):
    day[i] = val_labels[:,i]
    mae_base1[i] = np.nanmean(np.abs(preds - day[i]))
    print('normalized MAE of base model for day ', i+1, " is ", mae_base1[i])
    print('unnormalized MAE of base model for day ', i+1, " is ", mae_base1[i]*std[0])
mae_base_mean = mae_base1.mean()
print('mean of normalized MAE of base model of week ', " is ", mae_base_mean)
print('mean of unnormalized MAE of base model of week ', " is ", mae_base_mean*std[0])

"""Save your model:"""
#model.save('DAG_MultiOutput_1day')
