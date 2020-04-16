@author: aza8223
"""
###############################################################################
###############################################################################
###############################################################################
"""Deep NN - Project1 - Azad Almasov"""
###############################################################################
###############################################################################
###############################################################################




"""Importing models to be used"""

from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
import pandas as pd
from keras.utils.np_utils import to_categorical

"""Feature extraction function for Transfer learning"""
datagen = ImageDataGenerator(rescale=1./255)
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

"""Establish convolutional base model for transfer learning"""
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
###############################################################################
###############################################################################
###############################################################################


"""Data Processing"""
total_data = pd.read_csv("leafClassification/train.csv").values
total_data = total_data[:,:2]
#np.random.shuffle(train_data1) #I skipped shufflinf since my data already shuffled
train_data1 = total_data[:round(len(total_data)*0.8),:]
total_labels = np.unique(train_data1[:,1])

###############################################################################
###############################################################################
###############################################################################
"""K-fold Validation"""
k = 4 #I took k as 4 since my data is already very small. Some classes have
      #only 4 samples
num_val_samples = len(train_data1) // k
batch_size = 6
num_epochs = 100
all_scores = []
all_val_acc = []
all_train_acc = []


"""K-Fold Validation Starts"""
for bb in range(k):
    
    print('processing fold #', bb)

    total_data = pd.read_csv("leafClassification/train.csv").values
    total_data = total_data[:,:2]
    train_data1 = total_data[:round(len(total_data)*0.8),:]
    original_dataset_dir = 'leafClassification/images'
    base_dir = 'leaves'
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir,'train')
    validation_dir = os.path.join(base_dir,'validation')
    test_dir = os.path.join(base_dir,'test')
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
          
    """Train, validation data and labels:"""   
    val_data = train_data1[bb * num_val_samples: (bb + 1) * num_val_samples,:]
    train_data = np.concatenate(
                  [train_data1[:bb * num_val_samples,:],
                   train_data1[(bb + 1) * num_val_samples:,:]])
 
    
    total_labelss=[]
    """Assign indice to each class"""
    for i in enumerate(total_labels):
        total_labelss.append(i)

    total_labelss = dict([(value, key) for (key, value) in dict(total_labelss).items()])
    
    """Create folder for each classes:"""
    """Copying test and train data images. Further I will get validation data from
    training data"""

    for k,v in total_labelss.items():
        a = os.path.join(train_dir,k)
        os.mkdir(a)
        for i, j in train_data:
            if j in k:
                fname =  '{}.jpg'.format(i)
                src = os.path.join(original_dataset_dir,fname)
                dst = os.path.join(a,fname)
                shutil.copyfile(src, dst)
        print('total training ', k, ' data: ', len(os.listdir(a)))

    for k,v in total_labelss.items():
        a = os.path.join(validation_dir,k)
        os.mkdir(a)
        for i, j in val_data:
            if j in k:
                fname =  '{}.jpg'.format(i)
                src = os.path.join(original_dataset_dir,fname)
                dst = os.path.join(a,fname)
                shutil.copyfile(src, dst)
        print('total validation ', k, ' data: ', len(os.listdir(a)))

    """Assign those indices as label to our test, training and validation data
    as well as check if any label missing in them"""

    for i,j in enumerate(train_data[:,1]):
        if train_data[i,1] in list(total_labelss.keys()):
            train_data[i,1]=total_labelss[j]
        else:
            print(j, ' label is missing in training data')

    for k,v in enumerate(val_data[:,1]):
        if val_data[k,1] in list(total_labelss.keys()):
            val_data[k,1]=total_labelss[v]
        else:
            print(v, ' label is missing in validation data')
   
    train_labels = train_data[:,1]
    val_labels = val_data[:,1]

    """Vectorize labels"""
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)


    """TRAINING PROCESS"""


    """Feature Extraction"""
    train_features, train_labels = extract_features(train_dir, len(train_data))
    validation_features, validation_labels = extract_features(validation_dir, len(val_data))
    """Feature vectorization"""
    train_features = np.reshape(train_features, (len(train_data), 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (len(val_data), 4 * 4 * 512))

    """Building dense model"""
    modelPT = models.Sequential()
    modelPT.add(layers.Dense(512, activation='relu', input_dim=4 * 4 * 512))
    modelPT.add(layers.Dropout(0.5))
    modelPT.add(layers.Dense(99, activation='softmax'))
    modelPT.summary()

    """Compiling"""
    modelPT.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])
    """Training"""
    historyPT = modelPT.fit(train_features, train_labels,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels),
                    verbose=0)
    
    """Save your training and valiation accuracies"""
    validation_acc = historyPT.history['val_acc']
    all_val_acc.append(validation_acc)      

    training_acc = historyPT.history['acc']
    all_train_acc.append(training_acc) 
    
    """Delete all your folders and data before going to next iteration
    since it will give you error"""
    shutil.rmtree('leaves')
    del train_data
    del val_data
    del train_labels
    del val_labels
    del train_features
    del validation_features
    del total_data
    del train_data1
"""End of k-fold validation"""

"""Take average of training and validation accuracy by fold per each epoch"""
average_val_acc = [np.mean([x[i] for x in all_val_acc]) for i in range(num_epochs)]
average_train_acc = [np.mean([x[i] for x in all_train_acc]) for i in range(num_epochs)]

"""Plotting average val_acc and train_acc"""
import matplotlib.pyplot as plt
epochs = range(1, len(average_val_acc) + 1)
plt.plot(epochs, average_train_acc, 'bo', label='Training acc')
plt.plot(epochs, average_val_acc, 'b', label='Validation acc')
plt.title('k-foldTraining and validation accuracy')
plt.legend()
plt.figure()
plt.show()



###############################################################################
###############################################################################
#################################OOOOOO########################################
###############################################################################
###############################################################################

"""TRAINING WHOLE TRAINING DATA"""

original_dataset_dir = 'leafClassification/images'

base_dir = 'leaves'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

"""First I import datat classes names and picture labes so that i can create
appropriate folder for each of them"""
total_data = pd.read_csv("leafClassification/train.csv").values
total_data = total_data[:,:2]
#np.random.shuffle(total_data) #I skipped shufflinf since my data already shuffled
train_data = total_data[:round(len(total_data)*0.8),:]
test_data = total_data[round(len(total_data)*0.8):,:]
train_indices = train_data[:,0]
test_indices = test_data[:,0]

"""Get rid of repeated labels so that to get total number of classes (99)
in alphabetical order"""
total_labels = np.unique(train_data[:,1])
total_labelss=[]

"""Assign indice to each class"""
for i in enumerate(total_labels):
    total_labelss.append(i)

total_labelss = dict([(value, key) for (key, value) in dict(total_labelss).items()])


"""Create folder for each classes:"""
"""Copying test and train data images"""
for k,v in total_labelss.items():
    a = os.path.join(train_dir,k)
    os.mkdir(a)
    for i, j in train_data:
        if j in k:
            fname =  '{}.jpg'.format(i)
            src = os.path.join(original_dataset_dir,fname)
            dst = os.path.join(a,fname)
            shutil.copyfile(src, dst)
    print('total training ', k, ' data: ', len(os.listdir(a)))

            
for k,v in total_labelss.items():
    a = os.path.join(test_dir,k)
    os.mkdir(a)
    for i, j in test_data:
        if j in k:
            fname =  '{}.jpg'.format(i)
            src = os.path.join(original_dataset_dir,fname)
            dst = os.path.join(a,fname)
            shutil.copyfile(src, dst)
    print('total test ', k, ' data: ', len(os.listdir(a)))

"""Assign those indices as label to our test, training and validation data
as well as check if any label missing in them"""
for i,j in enumerate(train_data[:,1]):
    if train_data[i,1] in list(total_labelss.keys()):
         train_data[i,1]=total_labelss[j]
    else:
         print(j, ' label is missing in training data')
   
for k,v in enumerate(test_data[:,1]):
    if test_data[k,1] in list(total_labelss.keys()):
         test_data[k,1]=total_labelss[v]
    else:
         print(v, ' label is missing in validation data')


"""Training and test data labels:"""   
train_labels = train_data[:,1]
test_labels = test_data[:,1]

"""Vectorize labels"""
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


"""Feature Extraction"""
train_features, train_labels = extract_features(train_dir, len(train_data))
test_features, test_labels = extract_features(test_dir, len(test_data))
train_features = np.reshape(train_features, (len(train_data), 4 * 4 * 512))
test_features = np.reshape(test_features, (len(test_data), 4 * 4 * 512))

"""Building Model"""
modelPT = models.Sequential()
modelPT.add(layers.Dense(512, activation='relu', input_dim=4 * 4 * 512))
modelPT.add(layers.Dropout(0.5))
modelPT.add(layers.Dense(99, activation='softmax'))
modelPT.summary()

"""Compiling Model"""
modelPT.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

"""Training Model"""
historyPT = modelPT.fit(train_features, train_labels,
                    epochs=100,
                    batch_size=batch_size)

"""Saving the model"""
modelPT.save('modelPT_512')

"""Prediction"""
[losss, accur]  = modelPT.evaluate(test_features, test_labels)
print("Accuracy of the model is  ", accur)   

###############################################################################
###############################################################################
##############################VISUALIZATIONS###################################
###############################################################################
###############################################################################
###############################################################################
                            
"""Since I have used transfer learning, then my convolutional part is VG16,
therefore, Filter visualization will give exactly the same images as in the
book. Therefore, here, to be different, I will show results of my simpe model,
which gave 75 percent accuracy,-model 6 for visualization of filters and 
activations."""

"""VISUALIZING INTERMEDIATE ACTIVATIONS"""

from keras.models import load_model
model = load_model('model6')
model.summary()

img_test_ex_dir = 'leaves/test/Acer_Capillipes'
ex = os.listdir(img_test_ex_dir)

img_path = os.path.join(img_test_ex_dir,ex[0])


from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)


import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')

third_layer_activation = activations[2]
plt.matshow(third_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(third_layer_activation[0, :, :, 7], cmap='viridis')
plt.matshow(third_layer_activation[0, :, :, 30], cmap='viridis')


"""Visualizing every channel in every intermediate activation"""

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
            scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
"""VISUALISING CONVNET FILTERS"""

from keras.applications import VGG16
from keras import backend as K
#model = VGG16(weights='imagenet',
#              include_top=False)

from keras.models import load_model
model = load_model('model6')
model.summary()

layer_name = 'conv2d_90'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, model.input)[0]
iterate = K.function([model.input], [loss, grads])
import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

step = 1.
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    
    input_img_data += grads_value * step
    

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
#    x = np.clip(x, 0, 1)
#    x *= 255
#    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    
    return deprocess_image(img)

plt.imshow(generate_pattern('conv2d_90', 0))

"""Generating a grid of all filter response patterns in a layer"""
layer_name = 'conv2d_90'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img
                
plt.figure(figsize=(20, 20))
plt.imshow(results)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

"""VISUALISING HEATMAPS OF CLASS ACTIVATION"""
"""However, to visualize heatmap I used my transfer learning model which is
my main model"""
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')                        

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
                            
acer_capilipes_output = model.output[:, 386]           
last_conv_layer = model.get_layer('block5_conv3')
  
grads = K.gradients(acer_capilipes_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
                            
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
                            
import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
"""Save the superimposed image into the following directory - img_path"""
cv2.imwrite(img_path, superimposed_img)                            
                            
###############################################################################
###############################END#############################################
###############################################################################                            
 
