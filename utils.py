__author__ = 'Prateek'

'Import libraries'
import numpy as np
import math
from collections import Counter
import operator


"Gives the count of classes in the data provided"
def classDistribution(label):
    '''
    :param label: list containing true label of each data point
    :return: class and class distribution
    '''
    return np.unique(label,return_counts=True)


'Calculate the gini index'
def calGini(label):
    '''
    :param label: list containing true label of each data point
    :return: gini score
    '''
    gini = 0
    classes,counts = classDistribution(label)
    numSamples = label.shape[0]

    for c in counts:
        gini += np.square(c/numSamples)
    return 1- gini


'Calculate entropy'
def calEntropy(label):
    '''
    :param label: list containing true label of each data point
    :return: entropy score
    '''
    entropy = 0
    numSamples = label.shape[0]
    classes,count = classDistribution(label)
    for c in count:
        prob = c/numSamples
        entropy = entropy - prob * math.log(prob, 2)

    return entropy

'Calculate weighted entropy'
def calWeightedEntropy(label, weights):
    '''
    :param label: label of data points
    :param weights: weights of data points
    :return: entropy
    '''

    assert label.shape == weights.shape, 'Number of weights is not equal to number of data points.'
    totalsum = np.sum(weights)
    entropy = 0
    classes = np.unique(label)
    for c in classes:
        getindex = np.where(label == c)[0]
        prob = np.sum(weights[getindex])/totalsum
        entropy = entropy - prob * math.log(prob, 2)
    return entropy


"Divide data into two parts"
def divideData(data,featValue):
    '''
    :param data: data of a single column
    :param featValue: feature value used to split the data
    :return: indices of points belonging to left or right child
    '''
    leftind = np.where(data >= featValue)[0]
    rightind = np.where(data < featValue)[0]

    return leftind.astype(int),rightind.astype(int)

'Return the majority class'
def majClass(label):
    '''
    :param label: list containing true label of each data point.
    :return: the majority class label.
    '''
    assert label.ndim == 1, "1D array required"

    count = np.bincount(label)
    return np.argmax(count)

'Calculate pessimistic error'
def calPessimisticError(label):
    '''
    :param label: list containing true label of each data point
    :return: pessimistic error of adding a node.
    '''
    numSamples = label.shape[0]
    counter = Counter(label)
    majclass = max(iter(counter.items()), key=operator.itemgetter(1))[0]
    err = ((numSamples - counter[majclass]) + 2 * 0.5) / numSamples
    return err