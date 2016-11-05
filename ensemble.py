import time
import math
import numpy as np
import copy
from decisiontree import DecisiontreeClassifier
from multiprocessing import Process, Queue

class BaggingClassifier():
    '''
    Bagging classifier is meta-algorithm that builds a number of estimators on bootstrapped(with replacement)
    versions of the training dataset. Bagging is used on estimators which have high variance like a decision
    tree that has memorized the data i.e. there is a tree path for each data point. The prediction is done
    using various combination functions like weighted mean, average, max, min etc.
    '''

    def __init__(self, baseEstimator=None, n_estimators=10, bootstrap=True, random_state=None, max_depth=None):
        '''
        :param baseEstimator: the estimator to be used(default: Decision Tree)
        :param n_estimators: number of estimators to be used
        :param bootstrap: create bootstrap sample of data set(default True)
        :param random_state: random seed
        :param max_depth: max depth of the decision tree (default is None)
        '''
        self.baseEstimator = baseEstimator
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.Classifiers = Queue()
        self.max_depth = max_depth

    def fit(self, Xtrain, ytrain):
        start = time.time()

        # set the random seed for reproducibility of experiment
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Check if base estimator is specified otherwise use decision tree
        if self.baseEstimator is None:
            self.baseEstimator = DecisiontreeClassifier(max_depth=self.max_depth, usePes=False)

        'Start multiprocessing'
        jobs = []
        for i in range(5):
            'Generate random indices with replacement'
            ind = np.random.choice(a=Xtrain.shape[0], size=Xtrain.shape[0], replace=True)

            'Pass the bootstrapped dataset to the process'
            p = Process(target=self._fitparallel, args=(Xtrain[ind, :], ytrain[ind, :]))
            jobs.append(p)
            p.start()
        'Join the processes'
        for proc in jobs:
            proc.join()

        # end = time.time()
        # print('Total time:', end - start)

    'Function that will be called by the proecess to train the decision tree.'
    def _fitparallel(self,Xtrain,ytrain):
        classifier = copy.copy(self.baseEstimator)
        classifier.fit(Xtrain, ytrain)
        'save the trained classifier in queue'
        self.Classifiers.put(classifier)



    def predict(self, Xtest):
        '''
        :param Xtest: test data
        :return: predictions
        '''
        multiplePred = []
        pred = []

        for _ in range(self.Classifiers.qsize()):
            multiplePred.append(self.Classifiers.get().predict(Xtest))
        for j in range(Xtest.shape[0]):
            singlepred = []
            for i in range(len(multiplePred)):
                singlepred.append(multiplePred[i][j])
            pred.append(max(set(singlepred), key=singlepred.count))
        return np.array(pred)


class AdaboostClassifier():
    '''
    Adaboost classifier uses number of estimators with high bias like a decision tree stump
    and fits them on the dataset such that the subsequent estimators build on the mistakes
    of the previous estimators by increasing the weight of the samples which were incorrectly
    classified. This can be achieved by two methods:
    1. Sample more of the incorrectly classified data points and learn an estimator on them,
    but the error is calculated using the whole data set.
    2. Use a estimator than can handle sample weights like a decision tree which calculates
    weighted information gain.
    '''

    def __init__(self, baseEstimator=None, n_estimators=10, random_state=None, max_depth=1, useSampling=False,
                 verbose=False):
        '''
        :param baseEstimator: the estimator to be used(default: Decision Tree)
        :param n_estimators: number of estimators to be used
        :param random_state: random seed
        :param max_depth: max depth of the decision tree (default is decision stump)
        :param useSampling: Used to sample more of the incorrect predicted points(default False) Not implemented yet
        :param verbose: used to print out the values while calculating(default False)
        '''
        self.baseEstimator = baseEstimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.Classifiers = []
        self.max_depth = max_depth
        self.useSampling = useSampling
        self.verbose = verbose

    def fit(self, Xtrain, ytrain):
        '''
        :param Xtrain: training data
        :param ytrain: training labels
        :return: None
        '''
        # set the random seed for reproducibility of experiment
        if self.random_state != None:
            np.random.seed(self.random_state)

        # Check if base estimator is specified otherwise use decision tree
        if self.baseEstimator == None:
            self.baseEstimator = DecisiontreeClassifier(max_depth=self.max_depth)

        'Initialize the weights of data points to a uniform distribution'
        D = np.array([1 / (Xtrain.shape[0] * 1.0) for _ in range(Xtrain.shape[0])])

        for i in range(self.n_estimators):
            classifier = copy.copy(self.baseEstimator)

            if self.useSampling == True:
                ind = np.random.choice(a=Xtrain.shape[0], size=Xtrain.shape[0], p = D)
                'to be implemented'

            else:
                classifier.sampleWeights = D

            classifier.fit(Xtrain, ytrain)
            pred = np.array(classifier.predict(Xtrain))
            pred = pred.reshape((pred.shape[0], 1))

            weightedError = 0
            for j in range(Xtrain.shape[0]):
                if pred[j] != ytrain[j]:
                    weightedError += D[j]

            alpha = (1 / 2) * np.log((1 - weightedError) / (weightedError))

            if self.verbose == True:
                print('weighted error', weightedError)
                print('alpha:', alpha)


            self.Classifiers.append((classifier, alpha))
            for j in range(Xtrain.shape[0]):
                if pred[j] != ytrain[j]:
                    D[j] *= math.exp(alpha)
                else:
                    D[j] *= math.exp(-alpha)

            'Normalize the weights of data points'
            sumofWeights = np.sum(D)
            D = np.array([x / sumofWeights for x in D])

    def predict(self, Xtest):
        '''
        :param Xtest: test data
        :return: predictions {0,1}
        '''
        multiplePred = []
        pred = []
        for classifier, weight in self.Classifiers:
            multiplePred.append((classifier.predict(Xtest), weight))

        for j in range(Xtest.shape[0]):
            singlepred = 0
            for i in range(len(self.Classifiers)):
                singlepred += multiplePred[i][0][j] * multiplePred[i][1]
            if singlepred >= 0:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred)
