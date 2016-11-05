import utils
import time
import numpy as np

class _node():
    '''
    Node class that will define the node of a decision tree. It will store the feature index on which to split
    feature value which will be used to split, connection left child and right child and class label only if it
    is a leaf node.
    '''
    def __init__(self, featureIndex=-1,featureValue=None,leafnodeValue=None,leftBranch=None, rightBranch=None):
        '''
        :param featureIndex: store the value of best feature index
        :param featureValue: store the value of best feature
        :param leafnodeValue: class at the leaf node(None if not present)
        :param leftBranch: left child
        :param rightBranch: right child
        '''
        self._featureIndex =featureIndex
        self._featureValue= featureValue
        self._leafnodeValue = leafnodeValue
        self._leftBranch = leftBranch
        self._rightBranch = rightBranch


class DecisiontreeClassifier():
    '''
    Decision tree class which defines three functions viz fit and predict.
    fit: will learn a decision tree from data and save it.
    predict:  will be used to predict the labels for test data.
    '''

    def __init__(self,criteria = 'entropy',max_depth = None,usePes = False,sampleWeights = None,verbose=False):
        '''
        :param criteria: function to measure the quality of split(entropy/gini). Default is entropy.
        :param max_depth: max depth of the decision tree.
        :param usePes: use of pessimistic error to pre-prune the tree (True/False)
        :param sampleWeights: list containing weight of each sample in data.
        :param verbose: if true then the program will print the time taken to build the tree.
        '''
        self.dt = None
        self.criteria = criteria
        self.maxDepth = max_depth
        self.usePes = usePes
        self.sampleWeights = sampleWeights
        self.verbose = verbose


    def fit(self,Xtrain, ytrain):
        '''
        :param Xtrain: training data
        :param ytrain: training labels
        :return: None
        '''
        start = time.time()
        'Call the _builddecisiontree function to learn the tree'
        self.dt = self._builddecisiontree(Xtrain, ytrain,weights=self.sampleWeights)
        end = time.time()
        if self.verbose == True:
            print('Time taken to fit data:', end-start)

    def predict(self,Xtest):
        '''
        :param Xtest: test data
        :return: predicted values
        '''
        pred = []
        for row in Xtest:
            pred.append(self._evaluate(self.dt, row))
        return np.array(pred)



    def _builddecisiontree(self,Xtrain, ytrain, error = float('inf'),parentDepth=0,weights=None):
        '''
        :param Xtrain: train data
        :param ytrain: train label
        :param error: error of the parent node
        :param parentDepth: depth of the parent
        :param weights: weight of each sample
        :return: decision tree
        '''

        "condition for recursion to stop"
        if Xtrain.shape[0] == 0:
            return _node()
        if self.maxDepth != None:
            if self.maxDepth <= parentDepth:
                return _node(leafnodeValue=utils.majClass(ytrain))

        "Initalize child error"
        childError = error

        "Check if we have to use pessimistic error for pre-pruning"
        if self.usePes != False:
            "Calculate the data pessimistic error"
            errorPess = utils.calPessimisticError(label=ytrain)
            "Check if parent pessimistic error is less than child"
            if errorPess < error:
                childError = errorPess
            else:
                return _node(leafnodeValue=utils.majClass(ytrain))

        "Initalize variables"
        bestgain = 0
        bestsplit = None
        bestdivide = None


        if self.criteria == "gini":
            Algo = utils.calGini
        else:
            Algo = utils.calEntropy

        if self.sampleWeights != None:
            parentScore = utils.calWeightedEntropy(ytrain,weights)
        else:
            parentScore = Algo(ytrain)

        for featIndex in range(Xtrain.shape[1]):
            "Divide data based on feature value"
            for featValue in list(set(Xtrain[:,featIndex])):
                left, right = utils.divideData(Xtrain [:,featIndex],featValue)
                if self.sampleWeights != None:
                    leftent = utils.calWeightedEntropy(ytrain[left,:],weights[left])
                    rightent = utils.calWeightedEntropy(ytrain[right,:],weights[right])
                else:
                    leftent = Algo(label = ytrain[left,:])
                    rightent = Algo(label = ytrain[right,:])


                "Calculate the portion of data in left node"
                k = len(left) / (Xtrain.shape[0] * 1.0)
                gain = parentScore - k * leftent - (1 - k) * rightent
                if gain > bestgain and len(left) > 0 and len(right) > 0:
                    bestgain = gain
                    bestsplit = (featIndex, featValue)
                    bestdivide = (left, right)

        if bestgain > 0 and len(bestdivide[0]) > 0 and len(bestdivide[1])>0:
            if self.sampleWeights != None:
                leftweights = weights[bestdivide[0]]
                rightweights = weights[bestdivide[1]]
            else:
                leftweights = None
                rightweights = None


            leftBranch = self._builddecisiontree(Xtrain[bestdivide[0],:], ytrain[bestdivide[0],:],childError,(parentDepth + 1),leftweights)

            rightBranch = self._builddecisiontree(Xtrain[bestdivide[1],:], ytrain[bestdivide[1],:],childError,(parentDepth + 1),rightweights)

            return _node(featureIndex = bestsplit[0], featureValue = bestsplit[1], leftBranch = leftBranch,
                        rightBranch=rightBranch)
        else:
            return _node(leafnodeValue = utils.majClass(ytrain))


    "Evaluate the Decision Tree"
    def _evaluate(self, tree, data):
        '''
        :param tree: decision tree
        :param data: test data
        :return: predicted label of data
        '''
        if tree._leafnodeValue != None:
            return tree._leafnodeValue
        else:
            if data[tree._featureIndex] >= tree._featureValue:
                return self._evaluate(tree._leftBranch, data)
            else:
                return self._evaluate(tree._rightBranch, data)
