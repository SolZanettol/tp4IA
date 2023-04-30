"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""
import Prediction_Quali
import numpy as np

# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class DecisionTree():
	class Noeud:
		def __init__(self, niveau):
			self.niveau = niveau
			self.enfants = dict()

		def addChild(self, enfant, valeur):
			self.enfants[valeur] = enfant

		def getDernierNoeud(self, data):
			return self.enfants[data[self.niveau]].getDernierNoeud(data)

	class DernierNoeud:
		def __init__(self, label):
			self.label = label

		def getDernierNoeud(self, _):
			return self.label
		
	def train(self, train, train_labels):
		gains = self.calculAllGains(train, train_labels)
		self.racine = self.Noeud(np.where(gains==max(gains))[0][0])
		self.creerChildNoeuds(self.racine, train_labels, train)
		self.labels = np.unique(train_labels)

	def predict(self, x):
		return self.racine.getDernierNoeud(x)
	
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
		
	def creerChildNoeuds(self, parent, labels, data):
		for valeur in np.unique(data[parent.niveau]):
			filtre = data[parent.niveau] == valeur
			newLabels = labels[filtre]
			newData = data[:, filtre]
			unique = np.unique(newLabels)
			
			if len(unique) == 1:
				Noeud = self.DernierNoeud(newLabels[0])
			else:
				gains = self.calculAllGains(newData, newLabels)
				Noeud = self.Noeud(np.where(gains==max(gains))[0][0])
				self.creerChildNoeuds(Noeud, newLabels, newData)
			
			parent.addChild(Noeud, valeur)
		
	def calculAllGains(self, data, labels):
		labs, counts = np.unique(labels, return_counts=True)
		entropInit = self.calculEntropie(counts, len(labs))
		gains = [self.calculGain(catego, labels, entropInit) for catego in data]
		return np.array(gains)

	def calculGain(self, valeurs, labels, entropInit):
		unique, counts = np.unique(valeurs, return_counts=True)
		currentGain = 0
		for i in range(len(unique)):
			filtre = valeurs == unique[i]
			_, newLabelCounts = np.unique(labels[filtre], return_counts=True)
			entropie = self.calculEntropie(newLabelCounts, counts[i])
			currentGain = currentGain + (counts[i] / len(valeurs)) * entropie
		return entropInit - currentGain

	def calculEntropie(self, nombreLabels, denominateur):
		frEntrop = nombreLabels / denominateur
		return np.sum(-frEntrop * np.log2(frEntrop))