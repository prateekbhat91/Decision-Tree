import numpy as np
import math
import operator

"Gives the count of classes in the data provided"
def classDistribution(label):
    '''
    :param label: list containing true label of each data point
    :return: class distribution
    '''
    dist = {}
    for i in label:
        dist.setdefault(i[0],0)
        dist[i[0]] += 1
    return dist

'Calculate the gini index'
def calGini(label,weights):
    '''
    :param label: list containing true label of each data point
    :param weights: list containing weight of each sample
    :return: gini score
    '''
    gini = 0
    classdist = classDistribution(label)
    numSamples = label.shape[0]

    for classes in classdist:
        gini += pow((classdist[classes] / numSamples), 2)
    return 1 - gini


'Calculate entropy'
def calEntropy(label):
    '''
    :param label: list containing true label of each data point
    :return: entropy score
    '''
    entropy = 0
    numSamples = label.shape[0]
    classDist = classDistribution(label)
    for classes in classDist:
        prob = classDist[classes]/numSamples
        entropy = entropy - prob * math.log(prob, 2)
    return entropy

'Calculate weighted entropy'
def calWeightedEntropy(label, weights):
    '''
    :param label: lebel of data points
    :param weights: weights of data points
    :return: entropy
    '''
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
    left = []
    right = []
    for i in range(len(data)):
        if data[i] >= featValue:
            left.append(i)
        else:
            right.append(i)
    return left,right

'Return the majority class'
def majClass(label):
    '''
    :param label: list containing true label of each data point.
    :return: the majority class label..
    '''
    classDist = classDistribution(label)
    return max(iter(classDist.items()), key=operator.itemgetter(1))[0]

'Calculate pessimistic error'
def calPessimisticError(label):
    '''
    :param label: list containing true label of each data point
    :return: pessimistic error of adding a node.
    '''
    classDist = classDistribution(label)
    majclass = max(iter(classDist.items()), key=operator.itemgetter(1))[0]
    numSamples = label.shape[0]
    err = ((numSamples - classDist[majclass]) + 2 * 0.5) / numSamples
    return err