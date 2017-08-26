
from __future__ import print_function

import numpy as np
import tflearn
from tflearn import *
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
from sklearn.metrics import confusion_matrix
from tflearn.data_utils import to_categorical
from sys import argv
from tflearn.data_utils import load_csv

# The current model is trained on 7 different fields (Resource_List) to
# figure out the correct actual running time

# The argument has order as follows:
#   1. CSV file containing training data
filename = 'label_classifier.csv'
data, labels = load_csv(argv[1], categorical_labels = True, n_classes = 7)

print (labels,)

# Preprocess data (Normalize data)
data = np.array(data)
data = data.astype(float)
data_test =data
data -= np.mean(data)
data /= np.max(data)
labels = np.array(labels)
labels = labels.astype(float)
label_test = labels

# Build neural network
net = tflearn.input_data(shape=[None, 7])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)

net = tflearn.fully_connected(net,7,activation='softmax')
adam = Adam(learning_rate = 0.001, beta1 = 0.99)
#net = tflearn.regression(net,optimizer = adam, loss = 'mean_square', metric = R2())
net = tflearn.regression(net, optimizer = adam, learning_rate = 0.001,loss = 'categorical_crossentropy')

# Define model
model = tflearn.DNN(net, tensorboard_verbose = 0, tensorboard_dir ="logs",)

batch_size = 10
sub_data, sub_data_test, sub_label, sub_label_test = train_test_split(data, labels, test_size = 0.2, random_state = 0)

# Start training (apply gradient descent algorithm)
hist = model.fit(sub_data, sub_label, validation_set = (sub_data_test, sub_label_test), n_epoch=12, batch_size=batch_size, show_metric=True, shuffle = True)
model.save('resources_used_removed_classifer.tflearn')


score = model.evaluate(sub_data_test, sub_label_test, batch_size=batch_size)
print('Test score:', score[0])

dataTest, labelsTest = load_csv('test_Against_labelled.csv', categorical_labels = True, n_classes = 7)

