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
