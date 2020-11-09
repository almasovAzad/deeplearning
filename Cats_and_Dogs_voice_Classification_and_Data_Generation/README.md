### Statement of the Problem:
We have .WAV files for the voice of dogs and cats. The "train_test_split.exe" file shows the training and test data splitted. Our main objective in this project is to classify the voices of dogs and cats accurately.

### Solution:
Data preperation was very important part of this problem. WAV files we have are in int16 format. Therefore, we need to remember when we generate data we need to convert its type into int16. Our rate is 16000 data point for 1 sec. I converted voice into numerical data using "scipy.io.wavfile" function where you can read and write WAV files. First of all I need to pad them into the same length. After downloading them I have defined max-len=100000 for my data length. These WAV files are not pure cat and dog voices, they are contaminated with the background noise such as door opening. Therefore, I have defined portion to cut my data into that portion from different sections of the WAV file evenly skipping some points so that I have good generalization about voice; then, combined those small sections in order to get max-len data.
After preparetion of the data and splitting it into training, validation and test data sets using uniform sampling, I used different architecture of DNN algorithms. Then compared their accuracies.
Using well-trained model, I wrote code which takes output as its input and keeps generating data. By this way, I generated artificical voice of cats and dogs.

### Programs:
- Python (keras, numpy, pandas, scipy, matplotlib, random)
- Excel
- LaTex

### Conclusions:
1. Data preperation is very important and is problem specific.
2. Choosing appropriate optimization method in DNN algorithms is very important depending on problem, especially where your loss value directly stabilize very fast and you see overfitting at very early time.
3. In regression, depending on your output range, choosing appropriate scaling and normalizing can affect your results dramatically.

### Recommendation:
For voice generation I would also suggest to use variational autoencoders.
