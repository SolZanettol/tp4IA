import numpy as np
import sys
import load_datasets
import DecisionTree
from DecisionTree import DecisionTree # importer la classe de l'arbre de décision
import NeuralNet
from NeuralNet import NeuralNet# importer la classe du NN

from matplotlib import pyplot as plt
from sympy.ntheory import factorint

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""

# Hidden Layers validation
def xValidateHiddenLayers(dataset, name):
    hiddenSizes = [1, 4, 7, 10, 13, 16] #Pourquoi ceux-là?
    moyennes = [getMoyenneAccProf(dataset, hiddenSize, 1) for hiddenSize in hiddenSizes]
    plt.plot(hiddenSizes, moyennes)
    plt.savefig("../visuels/{filePrefix}-layers.png".format(filePrefix=name))

# Depth validation
def xValidateProfondeur(dataset, name, hiddenLayerSize):
    profs = [1,2,3,4,5]
    moyennes = [getMoyenneAccProf(dataset, hiddenLayerSize, prof) for prof in profs]
    plt.plot(profs, moyennes)
    plt.savefig("../visuels/{filePrefix}-depth.png".format(filePrefix=name))

# Params validation
def validateHyperparams(dataset, hiddenSize, prof):
    train, train_labels, test, test_labels = dataset
    batchSize = list(factorint(train.shape[0]))[-1]
    uniqueLabels = np.unique(train_labels)
    
    nn = NeuralNet(pLearningRate=0.05, pBatchSize=batchSize, pEpoque=100, pOutputSize=len(uniqueLabels), pInputSize=train.shape[1], pNHiddenLayers=prof, pHiddenSize=hiddenSize)
    nn.train(train, train_labels)
    return nn.evaluate(test, test_labels)

# Accuracy average
def getMoyenneAccProf(dataset, hiddenSize, prof):
    train, train_labels, _, _ = dataset
    
    step = 10
    data = train[:-(len(data) % step)]
    labels = train_labels[:-(len(labels) % step)]
    
    folds = []
    folds_labels = []
    for k in range(step):
        folds.append(data[k::step])
        folds_labels.append(labels[k::step])
    
    nbLabels= len(np.unique(labels))
    accuracies = []
    for index, (fold, fold_labels) in enumerate(zip(folds, folds_labels)):
        train = np.concatenate(folds[:index] + folds[index + 1:], axis = 0) if index else np.concatenate(folds[1:], axis = 0)
        train_labels = np.concatenate(folds_labels[:index] + folds_labels[index + 1:], axis = 0) if index else np.array(folds_labels[1:]).flatten()
        batchSize = list(factorint(train.shape[0]))[-1]
        nn = NeuralNet(pLearningRate=0.05, pBatchSize=batchSize, pEpoque=20, pOutputSize=nbLabels, pInputSize=train.shape[1], pNHiddenLayers=prof, pHiddenSize=hiddenSize)
        nn.train(train, train_labels)
        resultats = nn.evaluate(fold, fold_labels)
        accuracies.append([result.getInfo()[0] for result in resultats])

    return np.average(accuracies)

# Decision Tree tests
def decisionTreeTesting(DS, name):
    decisionTree = DecisionTree()
    decisionTree_SKLEARN = DecisionTreeClassifier(criterion = 'entropy')
    decisionTree.train(DS[0], DS[1])
    decisionTree_SKLEARN.fit(DS[0], DS[1])

    myPredictions = decisionTree.evaluate(DS[0], DS[1])
    skPredictions = decisionTree_SKLEARN.predict(DS[0])
    buildStr = ""
    for _, prediction in myPredictions:
        buildStr += str(prediction)
    print("\n\n-=[======================= Decision Tree Testing for {testName} DataSet (Training Data) =======================]=-\n-=[======================= My Decision Tree =======================]=-\n{mine}\n\n\n-=[======================= SciKit-Learn =======================]=-\n{sk}".format(testName=name, mine = buildStr, sk=confusion_matrix(skPredictions, DS[1])))
    
    myPredictions = decisionTree.evaluate(DS[2], DS[3])
    skPredictions = decisionTree_SKLEARN.predict(DS[3])
    buildStr = ""
    for _, prediction in myPredictions:
        buildStr += str(prediction)
    print("\n\n-=[======================= Decision Tree Testing for {testName} DataSet (Testing Data) =======================]=-\n-=[======================= My Decision Tree =======================]=-\n{mine}\n\n\n-=[======================= SciKit-Learn =======================]=-\n{sk}".format(testName=name, mine = buildStr, sk=confusion_matrix(skPredictions, DS[3])))

# Neural Network tests
def neuralNetTesting(DS, name):
    batchSize = list(factorint(train.shape[0]))[-1]
    nbLabels = len(np.unique(DS[1]))

    neuralNet = NeuralNet(pLearningRate=0.05, pBatchSize=batchSize, pEpoque=100, pOutputSize=nbLabels, pInputSize=train.shape[1], pNHiddenLayers=1, pHiddenSize=10)
    neuralNet_SKLEARN = MLPClassifier(hidden_layer_sizes = (10,), activation = 'logistic', batch_size = batchSize, learning_rate = 'constant', learning_rate_init = 0.05, max_iter = 100)
    neuralNet.train(DS[0], DS[1])
    neuralNet_SKLEARN.fit(DS[0], DS[1])

    myPredictions = neuralNet.evaluate(DS[0], DS[1])
    skPredictions = neuralNet_SKLEARN.predict(DS[0])
    buildStr = ""
    for _, prediction in myPredictions:
        buildStr += str(prediction)
    print("\n\n-=[======================= Neural Network Testing for {testName} DataSet (Training Data) =======================]=-\n-=[======================= My Decision Tree =======================]=-\n{mine}\n\n\n-=[======================= SciKit-Learn =======================]=-\n{sk}".format(testName=name, mine = buildStr, sk=confusion_matrix(skPredictions, DS[1])))
    
    myPredictions = neuralNet.evaluate(DS[2], DS[3])
    skPredictions = neuralNet_SKLEARN.predict(DS[2])
    buildStr = ""
    for _, prediction in myPredictions:
        buildStr += str(prediction)
    print("\n\n-=[======================= Neural Network Testing for {testName} DataSet (Testing Data) =======================]=-\n-=[======================= My Neural Network =======================]=-\n{mine}\n\n\n-=[======================= SciKit-Learn =======================]=-\n{sk}".format(testName=name, mine = buildStr, sk=confusion_matrix(skPredictions, DS[3])))


# Init datasets
IRIS_DS = load_datasets.load_iris_dataset(70)
WINE_DS = load_datasets.load_wine_dataset(70)
ABALONE_DS = load_datasets.load_abalone_dataset(70)

# Data for report
datasets = [IRIS_DS, WINE_DS, ABALONE_DS]
testCases = ["Iris", "Wine", "Abalone"]
hSize = [7, 1 ,3]

# Test cases
for i in range(len(testCases)):
    print("Test case : {ds}\tHyperParams : {result}".format(ds=testCases[i], result=validateHyperparams(datasets[i], hSize[i], 1)))
    xValidateHiddenLayers(datasets[i], testCases[i])
    xValidateProfondeur(datasets[i], testCases[i], hSize[i])
    decisionTreeTesting(datasets[i], testCases[i])
    decisionTreeTesting(datasets[i], testCases[i])