# HW6 - "step 0" following Jay Alammar's tutorial, 
# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
# including some code from there

import numpy as np
import pandas as pd
import torch
import transformers as ppb
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import nltk

# Location of SST2 sentiment dataset
SST2_LOC = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv'
WEIGHTS = 'distilbert-base-uncased'
# Performance on whole 6920 sentence set is very similar, but takes rather longer
SET_SIZE = 2000

# Download the dataset from its Github location, return as a Pandas dataframe
def get_dataframe():
    df = pd.read_csv(SST2_LOC, delimiter='\t', header=None)
    return df[:SET_SIZE]

# Extract just the labels from the dataframe
def get_labels(df):
    return df[1]

# Get a trained tokenizer for use with BERT
def get_tokenizer():
    return ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)

# Convert the sentences into lists of tokens
def get_tokens(dataframe, tokenizer):
    return dataframe[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# We want the sentences to all be the same length; pad with 0's to make it so
def pad_tokens(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    return padded

# Grab a trained DistiliBERT model
def get_model():
    return ppb.DistilBertModel.from_pretrained(WEIGHTS)

# This step takes a little while, since it actually runs the model on all sentences.
# Get model with get_model(), 0-padded token lists with pad_tokens() on get_tokens().
# Only returns the [CLS] vectors representing the whole sentence, corresponding to first token.
def get_bert_sentence_vectors(model, padded_tokens):
    # Mask the 0's padding from attention - it's meaningless
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens).to(torch.int64), attention_mask=mask)
    # First vector is for [CLS] token, represents the whole sentence
    return word_vecs[0][:,0,:].numpy()

# General purpose scikit-learn classifier evaluator.  The classifier is trained with .fit()
def evaluate(classifier, test_features, test_labels):
    return classifier.score(test_features, test_labels)


########################################################################################################################
##################################################### Assignment 6 #####################################################
########################################################################################################################

# Setting up the dataframe and model
dataframe = get_dataframe()
tokenizer = get_tokenizer()
tokens = get_tokens(dataframe, tokenizer)
padded_tokens = pad_tokens(tokens)
model = get_model()
sentence_vectors = get_bert_sentence_vectors(model, padded_tokens)
labels = get_labels(dataframe)

# Number 1:
# find_closest_sentences: takes a list of BERT vectors and a list of corresponding sentences, and prints the two (different) sentences
# that are closest in the space by Euclidean distance.
def find_closest_sentences(vecs, sentences):
    min_dist = float("inf")
    sent_a = -1
    sent_b = -1
    for i in range(len(vecs) - 1):
        for j in range(i + 1, len(vecs)):
            dist = np.linalg.norm(vecs[j] - vecs[i])
            if dist < min_dist and sentences[i] != sentences[j]:
                min_dist = dist
                sent_a = i
                sent_b = j
    print(sentences[sent_a])
    print(sentences[sent_b])
# To run:
# find_closest_sentences(sentence_vectors, dataframe[0])

# Number 3:
# visualize_data:  performs PCA on the sentences' BERT vector representations
# and plots results
def visualize_data(vecs, labels):
    pca = IncrementalPCA(n_components = 2)
    pca.fit(vecs)
    vecs2d = pca.transform(vecs)
    x = []
    y = []
    for i in range(len(vecs)):
        x.append(vecs[i][0])
        y.append(vecs[i][1])
    plt.scatter(x, y, c=labels)
# To run:
# visualize_data(sentence_vectors, labels)

# Number 4:
train_features, test_features, train_labels, test_labels = train_test_split(sentence_vectors, labels)

# Trains and returns a Naive Bayes
# learner, where the features are assumed to be normally distributed in the space. This one is
# done for you as an example.
def train_gaussian_naive_bayes(train_features, train_labels):
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    return gnb

# Train and return an Adaboost learner with
# decision tree stump weak learners.
def train_adaboost(train_features, train_labels):
    adaboost = AdaBoostClassifier()
    adaboost.fit(train_features, train_labels)
    return adaboost

# "Train" k-nearest neighbors for k=5, and return the classifier.
def train_nearest_neighbors(train_features, train_labels):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features, train_labels)
    return knn

# Train a classic multilayer neural network ("multilayer perceptron" to scikit-learn) with logistic (sigmoid) activation function and
# 100 hidden units in a single hidden layer. Return the classifier.
def train_classic_mlp_classifier(train_features, train_labels):
    mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(100,))
    mlp.fit(train_features, train_labels)
    return mlp

# Not that deep, just 2 hidden layers of 100 units each, using the rectifier activation function. Return the classifier
def train_deep_mlp_classifier(train_features, train_labels): 
    mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(100,100))
    mlp.fit(train_features, train_labels)
    return mlp

# Train what's essentially a perceptron with a logistic (sigmoid) activation function; a linear method. (But it does use regularization,
# and in that sense is more sophisticated than a classic perceptron.)
def train_logistic_regression(train_features, train_labels):
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)
    return clf

gnb = train_gaussian_naive_bayes(train_features, train_labels)
adaboost = train_adaboost(train_features, train_labels)
knn = train_nearest_neighbors(train_features, train_labels)
classic_mlp = train_classic_mlp_classifier(train_features, train_labels)
deep_mlp = train_deep_mlp_classifier(train_features, train_labels)
lr = train_logistic_regression(train_features, train_labels)

# To print the evaluations:
# print(evaluate(gnb, test_features, test_labels))
# print(evaluate(adaboost, test_features, test_labels))
# print(evaluate(knn, test_features, test_labels))
# print(evaluate(classic_mlp, test_features, test_labels))
# print(evaluate(deep_mlp, test_features, test_labels))
# print(evaluate(lr, test_features, test_labels))

# Number 8:
# print(dataframe[0][11])

# Number 10:
nltk.download('averaged_perceptron_tagger')
sentence = dataframe[0][11]
sentence = "The most repugnant adaptation of a classic text since Roland Joff and Demi Moore's The Scarlet Letter"
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
# print(tagged)

# Number 11:
nltk.download('maxent_ne_chunker')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
# print(entities)

# Number 12:
def get_sentiment_and_NNPs(classifier, model, text):
    # Set up the tokens/sentence vectors to put into the BERT model
    BERT_tokens = tokenizer.encode(text, add_special_tokens=True)
    vecs = get_bert_sentence_vectors(model, np.array([BERT_tokens]))
    sentiment = classifier.predict(vecs)

    # Get the tags from the sentence tokens (using nltk)
    NLTK_tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(NLTK_tokens)
    # Create the list of lists of consecutive NNPs
    nnps = []
    consec = False
    for word in tagged:
        if word[1] == 'NNP' and consec:
            nnps[len(nnps) - 1].append(word[0])
        elif word[1] == 'NNP':
            nnps.append([word[0]])
            consec = True
        else:
            consec = False
    return sentiment[0], nnps
# To test get_sentiment_and_NNPs:
# sentiment, nnps = get_sentiment_and_NNPs(lr, model, sentence)
# print(sentiment)
# print(nnps)