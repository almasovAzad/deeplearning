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
