### Statement of the problem:
This project is classification of the images of the leaves using different types of deep neural network methods. Download the data from: https://www.kaggle.com/c/leaf-classification. The main objectives of this project are:
1. To investigate deep learning method that gives higher accuracy. Compare different deep learning methods: their accuracy as well as their efficiency.
2. Visualize activations and filters of convolutional neural network.

### Solution:
I divided the images as well as the excel files, containing intensity of the each pixel of the given images, into three sets: training, validation, and test sets. To make sure that each data set is balanced—roughly equal number of each class I have used stratified (uniform) sampling. Then I used each of the following techniques to create classifier model:
a. training a dense network directly on these data sets;
b. training a deep network (convolutional neural network (CNN)) directly on these data sets;
c. training a deep network directly on these data sets with data augmentation (minimize the use of excessive transformations);
d. using fine tuning with dense network;
e. using fine tuning with dense network with the last convolutional block.

Then I compared computational time and accuracy of these models to choose the optimum one. 
I also visualized the activations of each layer of the model due to any two classes of my choice.

### Programs:
- Python, keras library, tensor flow and libraries for data visualisation.
- Excel
- LaTex

### Conclusions:
1. Best model was Transfer learning with 4-fold validation (modelPT-WHOLE-512), giving 88 percent
accuracy. In this transfer learning VGG16 is used as convolutional base network.
2. Colormap of the activation functions of each node of each layer of the model showed that most important feature of the leaf to be distibguished from other leaved is its shape of the edge, especially front edge of the leaf.
