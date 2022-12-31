# Text Classification with Recurrent Neural Networks RNN

Recurrent Neural Networks (RNN) are a type of Neural Network where the output from the previous step is fed as input to the current step. RNN's are mainly used for Sentiment Analysis, Sequence Labeling, Speech tagging, etc. There are four commonly used types of RNNs and one of them is Many-to-one. Many-to-One is used when a single output is required from multiple input units or a sequence of them. It takes a sequence of inputs to predict a fixed output. Sentiment Analysis is a common example of this type of RNN. In order to do this we need to consider a sentence as a word sequence (many), then classify its label(one). That is process of many-to-one type model.

# Part A: Document Preprocessing
1. First, the ‘IMDB_Dataset.csv’ file has been read, and for each document:
    a.  All non-letter characters have been removed.
    b. The short words (length ≤ 2) have been removed.
    c. All stop words (e.g., ‘a’, ‘and’, ‘what’, …), given in file ‘stopwords.txt’, have been removed.
2. The data has been tokenized and converted to word sequences.
3. Padding is added to ensure that all the sequences have the same length (the max length has been considered).
4. Words of each sequence have been converted into numerical vectors (one-hot encoding vector has been used). 

# Part B: many-to-one model
1. A many-to-one RNN model has been built with simple RNN (Elman Network).
2.  Documents have been split into training and test data. (80% for the train).
3. The model has been trained and the Accuracy of test data has been reported.

# Dataset:
IMDB dataset having 5K movie reviews for natural language processing or
Text analytics and Labeled by sentiment (positive and negative).
