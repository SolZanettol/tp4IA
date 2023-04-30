import numpy as np

class Prediction_Quali:
	def __init__(self, name, **kwargs):
		self.label = name
		self.TruePositive = 0
		self.FalsePositive = 0
		self.TrueNegative = 0
		self.FalseNegative = 0


	def inc_TP(self):
		self.TruePositive+=1

	def inc_TN(self):
		self.TrueNegative+=1

	def inc_FP(self):
		self.FalsePositive+=1

	def inc_FN(self):
		self.FalseNegative+=1

	def getInfo(self):
		accuracy = (self.TruePositive+self.TrueNegative)/(self.TruePositive+self.FalsePositive+self.TrueNegative+self.FalseNegative)
		precision = self.TruePositive/(self.TruePositive+self.FalsePositive)
		recall = self.TruePositive/(self.TruePositive+self.FalseNegative)
		f1_score = 2 * ((precision*recall)/(precision+recall))
		confu = np.array([np.array([self.TruePositive, self.FalseNegative]),np.array([self.FalsePositive, self.TrueNegative])])
		return accuracy, precision, recall, f1_score, confu

	def __str__(self):
		acc, pre, rec, f1, con = self.getInfo()
		return "\n{Label}:\n\tTrue Positive: {TP}\n\tTrue Negative: {TN}\n\tFalse Postive: {FP}\n\tFalse Negative: {FN}\n\n\tAccuracy: {acc}\n\tPrecision: {pre}\n\tRecall: {rec}\n\tF1 Score: {f1}\n\tConfusion Matrix: \n{con}\n\n".format(Label=self.label, TP=self.TruePositive, TN=self.TrueNegative, FP=self.FalsePositive, FN=self.FalseNegative,acc=acc, pre=pre, rec=rec, f1=f1, con=con)

	def simplifiedOutput(self):
		acc, pre, rec, f1, _ = self.getInfo()
		return "\n{Label}:\n\tTrue Positive: {TP}\n\tTrue Negative: {TN}\n\tFalse Postive: {FP}\n\tFalse Negative: {FN}\n\n\tAccuracy: {acc}\n\tPrecision: {pre}\n\tRecall: {rec}\n\tF1 Score: {f1}\n\n".format(Label=self.label, TP=self.TruePositive, TN=self.TrueNegative, FP=self.FalsePositive, FN=self.FalseNegative,acc=acc, pre=pre, rec=rec, f1=f1)