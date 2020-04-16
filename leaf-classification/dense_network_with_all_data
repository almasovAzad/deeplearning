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
