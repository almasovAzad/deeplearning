"""Base case for each day and mean of mae"""
"""Here I took average of previous days as my predictor for the each day of the
next week. Therefore I have calculated mae for each day of the next week. To
be able to compare this mae with my models, since I predict them all together, and
therefore I have 1 mae for  model, I took average of all those mae in this base
model for each day and took mean of them. I will use this mean of mae of the days of
the next week to compare it with my models. However, at the last model, where
I use multiple output DAG model, I used mae of each day in my base model to compare
it with the loss of each day in that last model:"""
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



"""Base model2: This is just my own opinion, but I ll not compare my models with this model.
In the following base model2, I choose my target  not as each dy of next week but average
of them. So I found mae between average precipitation of previous days as predictor of
average precipitation. This result showed 10 percent of mae. Compared to the base
model given above it is higher but it doesnt show that this is good predictor of
each day of next week, but it is good model to predict average precipitation of the
next week:"""
preds = np.mean(val_data[:, :, 0], axis=1)
week_data = np.mean(val_labels[:,:],axis=1)
mae_base2 = np.nanmean(np.abs(preds - week_data))
print('normalized MAE of base2 model is ', mae_base2)
print('unnormalized MAE of base2 model is ', mae_base2*std[0])
