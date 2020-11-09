### Statement of the problem:
This project is from Kagle: https://www.kaggle.com/rtatman/did-it-rain-in-seattle-19482017. Daily weather history of the Seattle between 1948 to 2017 are provided in the Excel file. In this Excel file for each day four feature are given: precipitation amount, maximum and minimum temperature, and logical variable indicating if it rained at that day or not. Our goal is to predict the weather of next day; next 7 days; next 14 days; and next 30 days accurately enough using different architecture of deep learning methods and compare their performance by means of accuracy and efficiency.

### Solution:
In this project, I used different architecture of DNN (deep neural network) methods to predict the weather:
1. First naive base case is created using mean absolute error.
2. Dense connected model is used.
3. Gated reccurent unit (GRU).
4. GRU with regularizer and dropout layer.
5. Bi-directional Long/short term memory (bi-directional LSTM).
6. LSTM with reversed sequence.
7. Convolutional neural network (CNN) with 1D dimension.
8. Combining CNN and GRU with long sequences.
9. Directed acyclic graph (DAG) network. The architecture of DAG is shown in the power point presentation file in this folder.
Accuracies of each model is compared with each other and specially with base case to compare the performance of the models.

### Programs:
- Python (keras, numpy, matplotlib, pandas, math, itertools, random)
- Excel
- Jupyter notebook is used to write report.

### Results:
Results are reported using Jupyter notebook. For each prediction problem (predicting 1 day; predicting; 7 days; predicting 14 days; predicting 30 days) I wrote seperate reports. And each report is saved under the folder with the corresponding name. In each folder, the best model for that corresponding problem were given as well.

### Conclusions:
Results show that for different prediction problems we need different kind of architecture of DNN models.

