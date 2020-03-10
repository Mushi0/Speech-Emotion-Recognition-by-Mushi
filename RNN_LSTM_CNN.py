# LSTM
import sys
import numpy as np
import keras
from keras import Sequential
from keras import layers
# from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, \
	BatchNormalization, Activation, MaxPooling2D, Embedding, SimpleRNN
from Common_Model import Common_Model
from Utils import plotCurve

class DNN_Model(Common_Model):
	# input_shape: the shape of input
	# num_classes: number of classes
	def __init__(self, input_shape, num_classes, **params):
		super(DNN_Model, self).__init__(**params)
		self.input_shape = input_shape
		self.model = Sequential()
		self.make_model()
		self.model.add(Dense(num_classes, activation = 'softmax'))
		self.model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		print(self.model.summary(), file = sys.stderr)
	
	# weights saved in Models as model_name.h5 and model_name.json
	def save_model(self, model_name):
		h5_save_path = 'Models/' + model_name + '.h5'
		self.model.save_weights(h5_save_path)
		
		save_json_path = 'Models/' + model_name + '.json'
		with open(save_json_path, "w") as json_file:
			json_file.write(self.model.to_json())
	
	# x_train, y_train: traing samples
	# x_val, y_val: test samples
	def train(self, model_name, x_train, y_train, x_val = None, y_val = None, n_epochs = 50):
		acc = []
		loss = []
		val_acc = []
		val_loss = []
		
		if x_val is None or y_val is None:
			x_val, y_val = x_train, y_train
		for i in range(n_epochs):
			p = np.random.permutation(len(x_train))
			x_train = x_train[p]
			# print(y_train.shape)
			y_train = y_train[p]
			
			# print(x_train.shape)
			'''if(model_name == 'cnn'):
				x_train = np.reshape(x_train, (1, x_train.shape[0], x_train.shape[1], 1))
				x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1], 1))'''
			history = self.model.fit(x_train, y_train, batch_size = 32, epochs = 1)
			# loss and accuracy on training dataset
			acc.append(history.history['accuracy'])
			loss.append(history.history['loss'])
			# loss and accuracy on test dataset
			val_loss_single, val_acc_single = self.model.evaluate(x_val, y_val)
			val_acc.append(val_acc_single)
			val_loss.append(val_loss_single)
		
		if(model_name == 'lstm'):
			plotCurve(acc, val_acc, 'LSTM Accuracy', 'acc')
			plotCurve(loss, val_loss, 'LSTM Loss', 'loss')
		elif(model_name == 'cnn'):
			plotCurve(acc, val_acc, 'CNN Accuracy', 'acc')
			plotCurve(loss, val_loss, 'CNN Loss', 'loss')
		elif(model_name == 'rnn'):
			plotCurve(acc, val_acc, 'RNN Accuracy', 'acc')
			plotCurve(loss, val_loss, 'RNN Loss', 'loss')
		self.trained = True
	
	def predict(self, sample):
		if not self.trained:
			sys.stderr.write("No Model.")
			sys.exit(-1)
		
		return np.argmax(self.model.predict(sample), axis = 1)
	
	def make_model(self):
		raise NotImplementedError()

class RNN_Model(DNN_Model):
	def __init__(self, **params):
		params['name'] = 'RNN'
		super(RNN_Model, self).__init__(**params)
	
	def make_model(self):
		self.model.add(SimpleRNN(128, input_shape = (1, self.input_shape), return_sequences=True))
		self.model.add(SimpleRNN(128, return_sequences=True))
		self.model.add(SimpleRNN(128))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(32,activation='relu'))

class LSTM_Model(DNN_Model):
	def __init__(self, **params):
		params['name'] = 'LSTM'
		super(LSTM_Model, self).__init__(**params)
	
	def make_model(self):
		self.model.add(KERAS_LSTM(128, input_shape = (1, self.input_shape)))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(32, activation = 'relu'))
		# self.model.add(Dense(16, activation='tanh'))

class CNN_Model(DNN_Model):
	def __init__(self, **params):
		params['name'] = 'CNN'
		super(CNN_Model, self).__init__(**params)
	
	def make_model(self):
		# Makes a CNN keras model with the default hyper parameters.
		self.model.add(Conv2D(8, (13, 13),
								input_shape=(
								self.input_shape[0], self.input_shape[1], 1)))
		self.model.add(BatchNormalization(axis = -1))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(8, (13, 13)))
		self.model.add(BatchNormalization(axis = -1))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size = (2, 1)))
		self.model.add(Conv2D(8, (13, 13)))
		self.model.add(BatchNormalization(axis = -1))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(8, (2, 2)))
		self.model.add(BatchNormalization(axis = -1))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size = (2, 1)))
		self.model.add(Flatten())
		self.model.add(Dense(64))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.2))
