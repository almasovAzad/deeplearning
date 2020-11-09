# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:50:18 2018

@author: Azad_Almasov
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 02:54:13 2018

@author: Azad_Almasov
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:57:35 2018

@author: Azad_Almasov
"""

###############################################################################
"""Project3: Cats and Dogs voice Classification and Data Generation"""
###############################################################################


"""Importing importatnt pakages"""
import numpy as np
import pandas as pd
import scipy.io.wavfile as sci_wav  # Open wav files
import matplotlib.pyplot as plt
import random
from keras.preprocessing import sequence
from keras import layers
from keras import optimizers
from keras.optimizers import RMSprop


"""Directories of audio files and train and test split files"""
voice_dir = 'cats_dogs'
csv_dir = 'train_test_split.csv' 


###############################################################################
"""Create functions to load and generate data"""
def read_wav_files(wav_files):
    '''Returns a list of audio waves
    Params:
        wav_files: List of .wav paths

    Returns:
        List of audio signals
    '''
    if not isinstance(wav_files, list): # If files are not as list make them list
        # The isinstance() function returns True if the specified object is of
        # the specified type, otherwise False.
        wav_files = [wav_files]
    return [sci_wav.read(voice_dir + '/' + f)[1] for f in wav_files]


"""Example wav data:"""
"""16000=1s)"""
acc = read_wav_files('dog_barking_43.wav')[0]
abc = range(1,len(acc)+1)
plt.figure()
plt.plot(abc, acc, 'b')
plt.legend()
plt.show()



"""Modify original data getting only values in some range for classification
problem. However dont do this for regression problem"""
"""Use less max_len value since you will already have less data due to range modification"""
df = pd.read_csv(csv_dir)
max_len=16000
portion_len=1600 # How many parts you wanna split data to feed it as sequential
"""NOTE: Your wav file are int16 so if you want to hear them again back you have to
return them back to int16 format"""
dataset = {}
# these are headers of our .CSV file
for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
        print(k)
        F=[]
        a = list(df[k].dropna()) # DataFrame.dropna() removes missing values
        v = read_wav_files(a)
        
#         Choose only the values between confidence interval mean+0.1*std and mean-0.1*std
#         Because when I listened most of the audios only at certain moments there
#         are cat or dog voice where amplitude goes high.
        ab = []
        for i,j in enumerate(v):
            mean = v[i].mean()
            std = v[i].std()
            ab.append(np.asarray([j[d] for d,l in enumerate(j) if mean-0.5*std>j[d] or j[d]>mean+0.5*std]))

        m = round(max_len/portion_len)
        asdf = np.zeros(shape=(len(a),portion_len,m)).astype('float32')
        for i, j in enumerate(ab):
            skip = round((len(ab[i])-max_len)/portion_len)
            if skip<0:
                skip = 0
            for n in range(portion_len):
                ab[i] = np.asarray(ab[i]).astype('float32')
                dim = ab[i][n*(m+skip):n*(m+skip)+m].shape[0]
                if dim == m:
                    asdf[i][n][:] = ab[i][n*(m+skip):n*(m+skip)+m]
                else:
                    asdf[i][n][:] = np.zeros(shape=(m))
        v = asdf
        v = np.reshape(v,(len(a),max_len))
        dataset[k] = v 

"""Example wav data:"""
"""16000=1s)"""
acc = dataset['train_cat'][74]
abc = range(1,len(acc)+1)
plt.figure()
plt.plot(abc, acc, 'b')
plt.legend()
plt.show()

"""Write WAV file: your data is 16000 hs. so you will have 16000 data in 1 sec"""
#filename = 'f444.wav'
#rate = 16000
#data = dataset['train_dog'][32]
#sci_wav.write(filename, rate, data)



train_data = np.concatenate((dataset['train_cat'],dataset['train_dog']), axis=0)
train_label = np.concatenate((np.ones((len(dataset['train_cat']))),np.zeros((len(dataset['train_dog'])))))

test_data= np.concatenate((dataset['test_cat'],dataset['test_dog']), axis=0)
test_label = np.concatenate((np.ones((len(dataset['test_cat']))),np.zeros((len(dataset['test_dog'])))))

# Compute mean and variance using all training data:
std, mean = train_data.std(), train_data.mean()

# Normalize your data
train_data = (train_data-mean)/std
test_data = (test_data-mean)/std

# Shuffle data
from random import shuffle

ind_list = [i for i in range(len(train_data))]
shuffle(ind_list)
train_data = train_data[ind_list,:]
train_label = train_label[ind_list]

ind_list = [i for i in range(len(test_data))]
shuffle(ind_list)
test_data = test_data[ind_list,:]
test_label = test_label[ind_list]

# Partition validation data from training data
val_data = train_data[0:round(len(train_data)/3),:]
val_label = train_label[0:round(len(train_data)/3)]

part_train_data = train_data[round(len(train_data)/3):,:]
part_train_label = train_label[round(len(train_label)/3):]




###############################################################################
###############################################################################
###############################################################################
"""Using Dense Network"""
from keras import regularizers
from keras.models import Sequential
model = Sequential()
model.add(layers.Dense(1024,kernel_regularizer=regularizers.l2(0.0001),activation='tanh'))
model.add(layers.Dense(32,kernel_regularizer=regularizers.l2(0.0001),activation='tanh'))
model.add(layers.Dense(1,activation='sigmoid'))


"""COMPILE YOUR MODEL"""
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
history = model.fit(part_train_data,
                    part_train_label,
                    epochs=epoch_size,
                    batch_size=batch_size,
                    validation_data=(val_data,val_label))


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
test_acc = model.evaluate(test_data, test_label)[1]
print('test_acc:', test_acc)

#"""Save your model:"""
#model.save('dense_model_3200dp0.5-320dp0.5-32dp0.5_REDIFINED')


###############################################################################
###############################################################################


"""Using RNN or GRU"""

"""Reshape your data for RNN"""

part_train_data_seq = np.reshape(part_train_data, (len(part_train_data),portion_len, round(max_len/portion_len)))
val_data_seq = np.reshape(val_data, (len(val_data),portion_len, round(max_len/portion_len)))
test_data_seq = np.reshape(test_data, (len(test_data),portion_len, round(max_len/portion_len)))


model = Sequential()
model.add(layers.GRU(4,
                     input_shape=(None, part_train_data_seq.shape[-1])))
model.add(layers.Dense(256,kernel_regularizer=regularizers.l2(0.001),activation='tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128,kernel_regularizer=regularizers.l2(0.005),activation='tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32,kernel_regularizer=regularizers.l2(0.01),activation='tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
    
    
"""COMPILE YOUR MODEL"""
#
#model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
#              loss='binary_crossentropy',
#              metrics=['acc'])

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              metrics=['acc'], optimizer=sgd)

"""TRAINING YOUR MODEL"""
epoch_size = 50
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
#model.save('gru_dp_0.2_dense32dp0.5_seq1')
    


###############################################################################
###############################################################################
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


###############################################################################
###############################################################################
###############################################################################
"""Conv1D""" """OPTIMIZER WAS IMPORTANT"""


model = Sequential()
model.add(layers.Conv1D(32, 3, activation='tanh',
                    input_shape=(None, part_train_data_seq.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 3, activation='tanh'))
model.add(layers.GlobalMaxPooling1D())#Global maxpooling gives you scalar output
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

sgd = optimizers.SGD(lr=0.015, decay=1e-6, momentum=0.9, nesterov=True)
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
#model.save('C1-32_mp_C1-32_gmp_dense256dp0.5-128dp0.5-32dp0.5_SGDoptim_REDIFINED')
    
###############################################################################
###############################################################################
###############################################################################


"""Learning sequential features"""

"""Use Original Data neglecting range modification you did for classification above:"""
"""Since you use your original data which will be long, use higher max-len value
than that in classification problem"""
df = pd.read_csv(csv_dir)
max_len=50000
portion_len=5000

dataset = {}
for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
        print(k)
        F=[]
        a = list(df[k].dropna()) # DataFrame.dropna() removes missing values
        v = read_wav_files(a)
        m = round(max_len/portion_len)
        asdf = np.zeros(shape=(len(a),portion_len,m)).astype('float32')
        for i, j in enumerate(v):
            skip = round((len(v[i])-max_len)/portion_len)
            if skip<0:
                skip = 0
            for n in range(portion_len):
                v[i] = np.asarray(v[i]).astype('float32')
                dim = v[i][n*(m+skip):n*(m+skip)+m].shape[0]
                if dim == m:
                    asdf[i][n][:] = v[i][n*(m+skip):n*(m+skip)+m]
                else:
                    asdf[i][n][:] = np.zeros(shape=(m))
        v = asdf
        v = np.reshape(v,(len(a),max_len))
        dataset[k] = v 

"""Split your data for new labels where is goind to be predicting next 10
amplitudes given 100 previous amplitudes:"""
train_dog_data = dataset['train_dog']
train_cat_data = dataset['train_cat']

test_dog_data = dataset['test_dog']
test_cat_data = dataset['test_cat']

#cat_or_dog = 1 # 1 for cat; 0 for dog; 2 for mixed; 
#if cat_or_dog==1:
#    print('training cat sequence')
#    train_data_conc = np.reshape(train_cat_data,(len(train_cat_data)*max_len,1))
#    test_data_conc = np.reshape(test_cat_data,(len(test_cat_data)*max_len,1))
#elif cat_or_dog==2:
#    print('training cat and dog mixed sequence')
#
#    train_data_conc = np.concatenate((train_dog_data,train_cat_data),axis=0)
#    test_data_conc = np.concatenate((test_dog_data,test_cat_data),axis=0)
#
#    ind_list = [i for i in range(len(train_data_conc))]
#    shuffle(ind_list)    
#    train_data_conc = train_data_conc[ind_list,:]
#    
#    ind_list = [i for i in range(len(test_data_conc))]
#    shuffle(ind_list) 
#    test_data_conc = test_data_conc[ind_list,:]
#    
#    train_data_conc = np.reshape(train_data_conc,(len(train_data_conc)*max_len,1))
#    test_data_conc = np.reshape(test_data_conc,(len(test_data_conc)*max_len,1))
#else:
#    print('training dog sequence')
#    train_data_conc = np.reshape(train_dog_data,(len(train_dog_data)*max_len,1))
#    test_data_conc = np.reshape(test_dog_data,(len(test_dog_data)*max_len,1))
train_data_conc = np.reshape(train_cat_data[0],(len(train_cat_data[0],1)))

# Normalize
#std = train_data_conc.std()
#mean = train_data_conc.mean()
#
#train_data_conc = (train_data_conc-mean)/std
#test_data_conc = (test_data_conc-mean)/std

minn = train_data_conc.min()
maxx = train_data_conc.max()

train_data_conc -= minn
train_data_conc /= (maxx-minn)

test_data_conc -= minn
test_data_conc /= (maxx-minn)

num_data = len(train_data_conc)
output_size = 10
input_size = 100


data_feat = np.zeros((num_data-(output_size+input_size),input_size,1))
data_label = np.zeros((num_data-(output_size+input_size),output_size))
test_data_feat = np.zeros((len(test_data_conc)-(output_size+input_size),input_size,1))
test_data_label = np.zeros((len(test_data_conc)-(output_size+input_size),output_size))
for i in range(num_data - (output_size+input_size)):
    data_feat[i] = train_data_conc[i:i+input_size]
    data_label[i] = train_data_conc[i+input_size:i+input_size+output_size,0]
for i in range(len(test_data_conc) - (output_size+input_size)):
    test_data_feat[i] = test_data_conc[i:i+input_size]
    test_data_label[i] = test_data_conc[i+input_size:i+input_size+output_size,0]  
    

val_feat_data = data_feat[:round(len(data_feat)/3)]
val_feat_label = data_label[:round(len(data_feat)/3)]

part_data_feat = data_feat[round(len(data_feat)/3):]
part_data_label = data_label[round(len(data_feat)/3):]


model = Sequential()
model.add(layers.Conv1D(32, 3, activation='relu',
                    input_shape=(None, part_data_feat.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 3, activation='relu'))
model.add(layers.GlobalMaxPooling1D())#Global maxpooling gives you scalar output
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(output_size, activation='relu'))

    
"""COMPILE YOUR MODEL"""
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='mae')

#sgd = optimizers.SGD(lr=0.015, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mae',
#              optimizer=sgd)


"""TRAINING YOUR MODEL"""
epoch_size = 6
batch_size = 1600
history = model.fit(part_data_feat,
                    part_data_label,
                    epochs=epoch_size,
                    batch_size=batch_size,
                    validation_data=(val_feat_data,val_feat_label))


"""Plotting results"""

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()   
    
"""PREDICTION - TESTING DATA"""
test_loss = model.evaluate(test_data_feat, test_data_label)
print('test_loss:', test_loss)
    
"""Save your model:"""
model.save('Normalized_and_z_scored_C1-32_mp_C1-32_gmp_dense256dp0.5-128dp0.5-32dp0.5_SGDoptim_REDIFINED_RELU_cat_split_100_10')



"""SEQUENTIAL DATA GENERATION"""

ns = 10000 # Number of sequence
seq = output_size*ns
from keras.models import load_model
model = load_model('Normalized_and_z_scored_gru4_DENSE64dp0.5-32_RELU_NOT-REDIFINED_cat_split_100_10')


arr = read_wav_files('cat_2.wav')[0].astype('float32')
arr -= mean
arr /=std
arr -=minn
arr /= (maxx-minn)
arr = np.reshape(arr,(len(arr),1))

test_data_arr = np.zeros((len(arr)-(output_size+input_size),input_size,1))

for i in range(len(arr) - (output_size+input_size)):
    test_data_arr[i] = arr[i:i+input_size]




a = model.predict(test_data_arr[:1,:,:])



asd = np.zeros(shape=(ns,output_size))
b = test_data_arr[:1:,:,:]
for i in range(ns):
    a = model.predict(b)
    b = np.concatenate((np.reshape(b,(b.shape[1])), np.reshape(a,(a.shape[1]))))
    b = b[output_size:]
    b = np.reshape(b,(1,input_size,1))
    c = np.reshape(a,(a.shape[1]))
    asd[i] = c

asd = np.reshape(asd,(ns*output_size)).astype('float32')
asd *=(maxx-minn)
asd +=minn
asd *=std
asd +=mean

asd = asd.astype('int16')
    
"""Plot resulted frequency"""
acc = asd
abc = range(1,len(acc)+1)
plt.figure()
plt.plot(abc, acc, 'b')
plt.legend()
plt.show()

"""Write WAV file: your data is 16000 hs. so you will have 16000 data in 1 sec"""
filename = 'f456.wav'
rate = 16000
data = asd
sci_wav.write(filename, rate, data)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

