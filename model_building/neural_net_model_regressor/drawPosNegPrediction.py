from __future__ import print_function

from tflearn import *
import tflearn
import numpy as np
import pandas as pd
from tflearn.data_utils import load_csv

from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# The order of argument is as follows:
#   1. CSV filename for training data
#   2. TFLearn training model
ZeroToTwo = {'Positive User Prediction':0, 'Positive Algorithm Prediction': 0, "Negative User Prediction": 0, "Negative Algorithm Prediction": 0}
TwoToFour = {'Positive User Prediction':0, 'Positive Algorithm Prediction': 0, "Negative User Prediction": 0, "Negative Algorithm Prediction": 0}
FourToSix = {'Positive User Prediction':0, 'Positive Algorithm Prediction': 0, "Negative User Prediction": 0, "Negative Algorithm Prediction": 0}
SixToEight = {'Positive User Prediction':0, 'Positive Algorithm Prediction': 0, "Negative User Prediction": 0, "Negative Algorithm Prediction": 0}
EightToTen = {'Positive User Prediction':0, 'Positive Algorithm Prediction': 0, "Negative User Prediction": 0, "Negative Algorithm Prediction": 0}
MoreThanTen = {'Positive User Prediction':0, 'Positive Algorithm Prediction': 0, "Negative User Prediction": 0, "Negative Algorithm Prediction": 0}

def userDecision(setName, user_diff):
    if (user_diff >= 0):
        setName['Positive User Prediction'] += 1
    else:
        setName['Negative User Prediction'] += 1

def algoDecision(setName, algo_diff):
    if (algo_diff >= 0):
        setName['Positive Algorithm Prediction'] += 1
    else:
        setName['Negative Algorithm Prediction'] += 1

def handleRange(runningTime, user_diff, algo_diff):
    if ((runningTime >= 0.0) and (runningTime <=2.0)):
        userDecision(ZeroToTwo, user_diff)
        algoDecision(ZeroToTwo, algo_diff)
    elif ((runningTime > 2.0) and (runningTime <= 4.0)):
        userDecision(TwoToFour, user_diff)
        algoDecision(TwoToFour, algo_diff)

    elif ((runningTime > 4.0) and (runningTime <=6.0)):
        userDecision(FourToSix, user_diff)
        algoDecision(FourToSix, algo_diff)

    elif ((runningTime > 6.0) and (runningTime <=8.0)):
        userDecision(SixToEight, user_diff)
        algoDecision(SixToEight, algo_diff)

    elif ((runningTime > 8.0) and (runningTime <=10.0)):
        userDecision(EightToTen, user_diff)
        algoDecision(EightToTen, algo_diff)

    elif ((runningTime > 10.0)):
        userDecision(MoreThanTen, user_diff)
        algoDecision(MoreThanTen, algo_diff)

def main():

    data, labels = load_csv(argv[1])

    data = np.array(data)
    data = data.astype(float)
    data_test =data
    data -= np.mean(data)
    data /= np.max(data)
    labels = np.array(labels)
    labels = labels.astype(float)
    label_test = labels

    net = tflearn.input_data(shape=[None, 13])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)

    net = tflearn.fully_connected(net,1,activation='relu')
    adam = Adam(learning_rate = 0.001, beta1 = 0.99)
    net = tflearn.regression(net,optimizer = adam, loss = 'mean_square', metric = R2())

    # Define model
    model = tflearn.DNN(net, tensorboard_verbose = 0)
    model.load(argv[2] + ".tflearn")

    prediction= model.predict(data_test)

    df = pd.read_csv('realTime_data.csv', usecols =['Resource_List.walltime'])
    userPredicted = np.array(df.values)
    userPredicted = userPredicted.astype(float)


    label_test = np.reshape(label_test, (-1,1))
    difference_algo = np.subtract(prediction, label_test)
    difference_user = np.subtract(userPredicted, label_test)

    acc = 0

    for x in np.nditer(difference_algo):
        if (x >= 0):
            acc += 1

    print ("Algorithm Accuracy is " + "{:.9f}".format(float (acc/float(difference_algo.size))))
    print ("Algorithm STD is " + "{:.9f}".format(float (np.std(difference_algo)/ 3600.0)))

    acc = 0
    for x in np.nditer(difference_user):
        if (x >= 0):
            acc += 1

    print ("User Accuracy is " + "{:.9f}".format(float (acc/float(difference_user.size))))
    print ("Standard Deviation is " + "{:.9f}".format(float (np.std(difference_user)/3600.0)))

    difference_algo /= 3600.0
    difference_user /= 3600.0


    for diff_user, diff_algo, actualRun in zip (difference_user, difference_algo, label_test):
        actualRun = actualRun/ 3600.0
        handleRange(actualRun, diff_user, diff_algo)

    myFig = plt.figure()
    ax = myFig.add_subplot(111)

    width = 0.75
    ind = np.arange(0,24,4)
    margin = 0.3

    rects1 = ax.bar(ind, np.array([ZeroToTwo['Positive User Prediction'],
                                   TwoToFour['Positive User Prediction'],
                                   FourToSix['Positive User Prediction'],
                                   SixToEight['Positive User Prediction'],
                                   EightToTen['Positive User Prediction'],
                                   MoreThanTen['Positive User Prediction']]),
                                    width = width, color = 'r', align = 'center')
    rects2 = ax.bar(ind + width + margin, np.array([ZeroToTwo['Positive Algorithm Prediction'],
                                                    TwoToFour['Positive Algorithm Prediction'],
                                                    FourToSix['Positive Algorithm Prediction'],
                             SixToEight['Positive Algorithm Prediction'],
                                                    EightToTen['Positive Algorithm Prediction'],
                             MoreThanTen['Positive Algorithm Prediction']]), width=width, color='b', align = 'center')

    rects3 = ax.bar(ind+ width *2 , np.array([ZeroToTwo['Negative User Prediction'],
                                              TwoToFour['Negative User Prediction'],
                                  FourToSix['Negative User Prediction'],
                                  SixToEight['Negative User Prediction'],
                                              EightToTen['Negative User Prediction'],
                                  MoreThanTen['Negative User Prediction']]), width=width, color='g', align = 'center')
    rects4 = ax.bar(ind + width * 3 ,
                    np.array([ZeroToTwo['Negative Algorithm Prediction'], TwoToFour['Negative Algorithm Prediction'],
                              FourToSix['Negative Algorithm Prediction'],
                              SixToEight['Negative Algorithm Prediction'], EightToTen['Negative Algorithm Prediction'],
                              MoreThanTen['Negative Algorithm Prediction']]), width=width, color='y', align='center')

    ax.set_ylabel('Frequency of occurrences')
    ax.set_xlabel('Time range')
    ax.set_xticks(ind + width)
    ax.set_title('Predictions vs Actual Running Time in terms of time range')
    ax.set_xticklabels(('0-2h','2h-4h', '4h-6h', '6h-8h', '8h-10h', '>10h'))
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Positive User Prediction', 'Positive Algorithm Prediction', 'Negative User Prediction','Negative Algorithm Prediction'))

# To label each column data (if necessary)

#    def autolabel(rects, message):
#        shift = 0
#        if (message == 'right'):
#            shift = 0.13
#
#        for rect in rects:
#            h = rect.get_height()
#            ax.text(rect.get_x() + rect.get_width() / 2. + shift, h + 5, '%d' % int(h),
#                    ha=message, va='bottom', fontsize = 8.5)

    # autolabel(rects1, 'center')
    # autolabel(rects2, 'center')
    # autolabel(rects3, 'center')
    # autolabel(rects4, 'center')
    myFig.savefig('running time predictions.png')
    plt.show()

if __name__ == "__main__":
	main()
