import numpy as np

def calAccuracy(ytest,pred):
    '''
    :param pred: vector containing all the predicted classes
    :param ytest: vector containing all the true classes
    :return: accuracy of classification
    '''
    #convert the list into 1-D arrays
    if len(ytest.shape) == 2:
        ytest = ytest.ravel()
    if len(pred.shape) == 2:
        pred = pred.ravel()

    count = 0
    for i,j in zip(pred,ytest):
        if i==j:
            count +=1
    return count/(1.0 * len(ytest))



def confusionMatrix(ytest,pred):
    '''
    :param ytest: test labels
    :param pred: predicted labels
    :return: confusion matrix
    '''
    if len(ytest.shape) == 2:
        ytest = ytest.ravel()
    if len(pred.shape) == 2:
        pred = pred.ravel()
    numclass = len(np.unique(pred))
    if numclass != 2:
        raise ValueError('can not handle multi-class problem as of now')

    confusion_matrix = np.array([[0,0],[0,0]])
    for p,e in zip(list(pred),list(ytest)):
        confusion_matrix[int(e)][int(p)] += 1
    return confusion_matrix


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




