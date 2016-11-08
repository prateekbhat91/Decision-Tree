__author__ = 'Prateek'

import numpy as np
from preprocessing import label_encoder


def convert_to_1D(array):
    '''
    Converts a numpy array into an array of 1 dimension.
    :param array: input numpy array
    :return: 1D array
    '''
    return np.ravel(array)



def calAccuracy(true,pred):
    '''
    :param true: vector containing all the true classes
    :param pred: vector containing all the predicted classes
    :return: accuracy of classification
    '''
    true = convert_to_1D(true)
    pred = convert_to_1D(pred)
    assert (true.shape == pred.shape), "true and pred dimensions do not match."
    return (true.shape[0] - np.count_nonzero(np.subtract(true, pred))) / true.shape[0]

def confusionMatrix(true,pred):
    '''
    :param true: numpy array containing all the true classes.
    :param pred: numpy array containing all the predicted classes.
    :return    : confusion matrix
    '''
    true = convert_to_1D(true)
    pred = convert_to_1D(pred)

    assert (true.shape == pred.shape), "true and pred dimensions do not match."
    numclass = len(np.unique(true))

    # encode the classes with integers ranging from 0 to numclass-1.
    labelEncoder = label_encoder()
    labelEncoder.fit(true)
    true = labelEncoder.transform(true)
    pred = labelEncoder.transform(pred)

    # create confusion matrix.
    # Rows indicate the true class and column indicate the predicted class.
    cm = np.array([np.zeros(numclass) for _ in range(numclass)])
    for t, p in zip(true, pred):
        cm[t][p] += 1
    return cm


def F_Score(ytest,pred):
    '''
        :param ytest: test labels
        :param pred: predicted labels
        :return: F-score
        '''
    cm = confusionMatrix(ytest, pred)
    numclass = cm.shape[0]
    if numclass != 2:
        raise ValueError('can not handle multi-class problem as of now')
    return (2 * cm[1][1]) / ((2 * cm[1][1] + cm[0][1] + cm[1][0]) * 1.0)


def printSummary(confusionMatrix,numData):
    correctlyClassified = sum(confusionMatrix[i][i] for i in range(len(confusionMatrix)))
    incorrectlyClassified = numData - correctlyClassified
    accuracy = correctlyClassified / numData

    print("\n=== Summary ===")
    print("Correctly Classified Instances = ", correctlyClassified)
    print("Incorrectly Classified Instances = ", incorrectlyClassified)
    print("Total Number of Instances = ", numData)
    print("Accuracy =", round((accuracy * 100), 2), "%")

def printconfusionMatrix(confusionMatrix):
    print("\n=== Confusion Matrix ===")
    print("Columns indicate the Predicetd values")
    print("Rows indicate the actual values")
    for rows in confusionMatrix:
        print(rows)




