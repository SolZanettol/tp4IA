import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')
    
    
    # TODO : le code ici pour lire le dataset
    
    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
    


#   Formatting dataset entries
    lines = [x.replace('\n', '') for x in f.readlines()]
    linesSplit = [x.split(',') for x in lines]
    random.shuffle(linesSplit)
    linesFormattedNoLabel = []
    for entry in linesSplit:
        linesFormattedNoLabel.append([float(x) for x in entry[0:-1]])

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (load_common_dataset(train_ratio, linesSplit, linesFormattedNoLabel))
	
	
	
def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le reste des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')

	
    # TODO : le code ici pour lire le dataset

#   Formatting dataset entries
    lines = [x.replace('\n', '') for x in f.readlines()]
    linesSplit = [x.split(',') for x in lines]
    random.shuffle(linesSplit)
    linesFormattedNoLabel = []
    for entry in linesSplit:
        linesFormattedNoLabel.append([float(x) for x in entry[0:-1]])

	# La fonction doit retourner 4 structures de données de type Numpy.
    return (load_common_dataset(train_ratio, linesSplit, linesFormattedNoLabel))

def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le reste des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    f = open('datasets/abalone-intervalles.csv', 'r') 

#   Formatting dataset entries
    lines = [x.replace('\n', '') for x in f.readlines()]
    linesSplit = [x.split(',') for x in lines]
    random.shuffle(linesSplit)
    linesFormattedNoLabel = []
    for entry in linesSplit:
        linesFormattedNoLabel.append([sex_to_int(entry[0])] + [float(x) for x in entry[1:-1]])

    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (load_common_dataset(train_ratio, linesSplit, linesFormattedNoLabel))

def sex_to_int(str):
    dic = {"I":0, "M":1, "F":2}
    return dic.get(str, -1) 

def load_common_dataset(train_ratio, plinesSplit, plinesFormattedNoLabel):
    """
        I created this function with the intention of increasing code readability and reduce code repetition.
        args : 
            train_ratio ->              This indicates the ratio of data to be put in the train and test sets respectively.
                                        Refer to the load_iris_dataset() method for further information.

            plinesSplit ->              Pass here the complete loaded dataset with the csv data in the form of a list. There 
                                        is no need to format this list.

            plinesFormattedNoLabel ->   Pass here the a formatted version of plinesSplit (They have to be ordered in the 
                                        same way). The label also needs to be absent from this list. Format the data in the
                                        appropriate way (string, float, int, etc) before passing it to the method.
    """
#   Validating train_ratio parameter
    if (train_ratio < 0 or train_ratio > 100):
        raise ValueError('Invalid Train Ratio number passed as argument, please specify a number between 0 and 100. This represents a percentage.')

#   Initialization of return variables
    train = []
    train_labels = []
    test = []
    test_labels = []

#   Important values to separate the dataset
    train_ratio/=100
    totalEntries = len(plinesSplit)
    breakPoint = (train_ratio * totalEntries)

#   Placing entries in their respective variables
    for i in range(totalEntries-1):
        if(i<breakPoint):
            train.append(plinesFormattedNoLabel[i])
            train_labels.append(plinesSplit[i][-1])
        else:
            test.append(plinesFormattedNoLabel[i])
            test_labels.append(plinesSplit[i][-1])

#   Transforming lists into Numpy Arrays
    train = np.array(train)
    train_labels = np.array(train_labels)
    test = np.array(test)
    test_labels = np.array(test_labels)

    return train, train_labels, test, test_labels