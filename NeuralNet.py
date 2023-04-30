"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""
from typing import Generator, Sequence
import Prediction_Quali
import numpy as np

# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class NeuralNet():
	class Neuron:
		def __init__(self, poidsInit):
			self.poidsInit = poidsInit

	class Couche:
		def __init__(self, previousSize, size):
			self.neurones = [NeuralNet.Neuron(np.random.random(previousSize)) for _ in range(size)]
			
		def activation(self, values):
			retour = (1 / (1 + np.exp(-values)))
			return retour

		def computeOutputs(self, input):
			output = np.zeros(len(self.neurones))
			for i in range(len(self.neurones)):
				output[i] = self.activation(sum(input * self.neurones[i].poidsInit))
			return output
	
		def calculPoidsDiff(self, diff):
			matPoids = np.stack([neurone.poidsInit for neurone in self.neurones])
			diff = diff.reshape(len(diff), 1)
			return np.sum(diff * matPoids, axis = 0)
	
		def backPropagate(self, learnRate, inputs, diff):
			grille = np.meshgrid(inputs, diff)
			poidsDiff =  learnRate * (grille[0] * grille[1])
			for i in range(len(inputs)):
				self.neurones[i].poidsInit += poidsDiff[i]

	def __init__(self, pLearningRate, pBatchSize, pEpoque, pOutputSize, pInputSize, pNHiddenLayers, pHiddenSize):
		self.couches = []
		self.learningRate = pLearningRate
		self.batchSize = pBatchSize
		self.epoque = pEpoque
		self.outputSize = pOutputSize

		previousSize = pInputSize
		for _ in range(pNHiddenLayers):
			self.couches.append(self.Couche(previousSize, pHiddenSize))
			previousSize = pHiddenSize
		self.couches.append(self.Couche(previousSize, pOutputSize))
		
	def train(self, train, train_labels):
		dataBatches = [train[i:i+self.batchSize] for i in range(0, len(train), self.batchSize)]
		labelBatches = [train_labels[i:i+self.batchSize] for i in range(0, len(train_labels), self.batchSize)]
		for _ in range(self.epoque):
			for dataBatch, labelBatch in zip(dataBatches, labelBatches):
				self.trainBatch(dataBatch, labelBatch)

	def trainBatch(self, data, labels):
		etapes = [None for _ in range(len(self.couches) + 1)]
		etapes[0] = data

		for index, couche in enumerate(self.couches):
			etapes[index + 1] = couche.computeOutputs(etapes[index])

		nextPoidsDiff = labels - etapes[-1]
		etapesInverses = list(enumerate(self.couches))[::-1]
		for index, couche in etapesInverses:
			diff = etapes[index + 1] * (1 - etapes[index + 1]) * nextPoidsDiff
			nextPoidsDiff = couche.calculPoidsDiff(diff)
			couche.backPropagate(self.learningRate, etapes[index], diff)

	def predict(self, x):
		for couche in self.couches:
			x = couche.computeOutputs(x)
		return np.where(x==max(x))[0][0]
	
	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		predicted = []
		for el in X:
			predicted.append(self.predict(el))
		predicted = np.array(predicted)

		qualifications = {x:Prediction_Quali.Prediction_Quali(x) for x in self.labels}

		for i in range(len(predicted)):
			prediction = predicted[i]
			correction = y[i]
			for qualit, obj in qualifications.items():
				if qualit == prediction and qualit==correction:
					obj.inc_TP()
				elif qualit != correction and qualit != prediction:
					obj.inc_TN()
				elif qualit != prediction and qualit==correction:
					obj.inc_FN()
				elif qualit == prediction and qualit!=correction:
					obj.inc_FP()
		return qualifications