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


