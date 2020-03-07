import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score

class Common_Model(object):
	def __init__(self, save_path, name = 'Not Specified'):
		self.model = None
		self.trained = False
	
	def train(self, x_train, y_train, x_val, y_val):
		raise NotImplementedError()
	
	def predict(self, samples):
		raise NotImplementedError()
	
	def predict_proba(self, samples):
		if not self.trained:
			sys.stderr.write("Not trained.")
			sys.exit(-1)
		return self.model.predict_proba(samples)
	
	def save_model(self, model_name):
		raise NotImplementedError()
	
	def evaluate(self, x_test, y_test):
		predictions = self.predict(x_test)
		print(y_test)
		print(predictions)
		print('Accuracy: %.3f\n' % accuracy_score(y_pred = predictions, y_true = y_test))
		
		'''
		predictions = self.predict(x_test)
		score = self.model.score(x_test, y_test)
		print("True Lable: ", y_test)
		print("Predict Lable: ", predictions)
		print("Score: ", score)
		'''

