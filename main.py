import numpy as np
from keras.utils import np_utils
import os

from MLP_SVM import SVM_Model, MLP_Model
from RNN_LSTM_CNN import RNN_Model, LSTM_Model, CNN_Model
# from RNN_LSTM_CNN import LSTM_Model
# from CNN_Model import CNN_Model
from VGGNet import VGG_Model

from Utils import load_model, Radar

import extractor as ex
from config import config

# the traing fuction
# model_name: the name of the model
# save_model_name: the name of the model to be saved with
# if_load: if the features has been loaded
def Train(model_name, save_model_name, if_load = True):
	if(model_name == 'cnn'):
		x_train, x_test, y_train, y_test, num_labels = ex.extract_data_cnn(flatten = False)
	
	# elif(model_name == 'vgg'):
	# 	x_train, x_test, y_train, y_test = ex.load_images(feature_path = config.IMAGES_PATH, train = True)
	
	elif(model_name == 'lstm2'):
		if(if_load == True):
			x_train, x_test, y_train, y_test = ex.load_feature(feature_path = config.TRAIN_FEATURE_PATH_NEW, train = True)
		else:
			x_train, x_test, y_train, y_test = ex.get_data_2(config.DATA_PATH, config.TRAIN_FEATURE_PATH_NEW, train = True)
	
	else:
		if(if_load == True):
			x_train, x_test, y_train, y_test = ex.load_feature(feature_path = config.TRAIN_FEATURE_PATH_LIBROSA, train = True)
		else:
			x_train, x_test, y_train, y_test = ex.get_data(config.DATA_PATH, config.TRAIN_FEATURE_PATH_LIBROSA, train = True)
	
	# build the model
	if(model_name == 'svm'):
		model = SVM_Model()
	elif(model_name == 'mlp'):
		model = MLP_Model()
	elif(model_name == 'rnn'):
		y_train = np_utils.to_categorical(y_train)
		y_val = np_utils.to_categorical(y_test)
		
		model = RNN_Model(input_shape = x_train.shape[1], num_classes = len(config.CLASS_LABELS))
		
		x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
		x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
	elif(model_name == 'lstm' or model_name == 'lstm2'):
		y_train = np_utils.to_categorical(y_train)
		y_val = np_utils.to_categorical(y_test)
		
		model = LSTM_Model(input_shape = x_train.shape[1], num_classes = len(config.CLASS_LABELS))
		
		x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
		x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
	elif(model_name == 'cnn'):
		y_train = np_utils.to_categorical(y_train)
		y_val = np_utils.to_categorical(y_test)
		
		in_shape = x_train[0].shape
		x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
		x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
		
		model = CNN_Model(input_shape = x_train[0].shape, num_classes = len(config.CLASS_LABELS))
	# elif(model_name == 'vgg'):
	# 	y_train = np_utils.to_categorical(y_train)
	# 	y_val = np_utils.to_categorical(y_test)
		
	# 	in_shape = x_train[0].shape
	# 	x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
	# 	x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
		
	# 	# print(x_train.shape)
	# 	model = VGG_Model(input_shape = x_train[0].shape, num_classes = len(config.CLASS_LABELS))
	
	print('-------------------------------- Start --------------------------------')
	if(model_name == 'svm' or model_name == 'mlp'):
		model.train(x_train, y_train)
	elif(model_name == 'rnn' or model_name == 'lstm' or model_name == 'cnn' or model_name == 'vgg' or model_name == 'lstm2'):
		model.train(model_name, x_train, y_train, x_test, y_val, 20)
	
	model.evaluate(x_test, y_test)
	model.save_model(save_model_name)
	print('---------------------------------- End ----------------------------------')
	
	return model

# prediction
# model: the model that has been trained
# model_name: the name of the model
# file_path: the path of files to be predicted
def Predict(model, model_name, file_path):
	# file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
	if(model_name == 'cnn'):
		test_feature = ex.get_feature_vector_from_mfcc(file_path, flatten = False)
		test_feature = test_feature.reshape(1, test_feature.shape[0], test_feature.shape[1], 1)
	# elif(model_name == 'vgg'):
	# 	test_feature = ex.imread(file_path)
	# 	test_feature = test_feature.reshape(1, test_feature.shape[0], test_feature.shape[1], 1)
	
	elif(model_name == 'lstm2'):
		test_feature = ex.get_data_2(train = False, feature_path = config.PREDICT_FEATURE_PATH_NEW, data_path = file_path)
	
	else:
		test_feature = ex.get_data(False, file_path, config.PREDICT_FEATURE_PATH_LIBROSA)
	
	if(model_name == 'rnn' or model_name == 'lstm' or model_name == 'lstm2'):
		test_feature = np.reshape(test_feature, (test_feature.shape[0], 1, test_feature.shape[1]))
	
	result = model.predict(test_feature)
	
	if(model_name == 'rnn' or model_name == 'lstm' or model_name == 'cnn' or model_name == 'vgg' or model_name == 'lstm2'):
		result = np.argmax(result)
	
	result_prob = model.predict_proba(test_feature)[0]
	print('Recogntion: ', config.CLASS_LABELS[int(result)])
	print('Probability: ', result_prob)
	Radar(result_prob)

# model = Train(model_name = 'mlp', save_model_name = 'MLP_LIBROSA', if_load = True)
# model = load_model(load_model_name = 'MLP_LIBROSA', model_name = 'mlp')
# Predict(model, model_name = 'mlp', file_path = 'Emo-db/test/test.wav')

# model = Train(model_name = 'svm', save_model_name = 'SVM_LIBROSA', if_load = True)
# model = load_model(load_model_name = 'SVM_LIBROSA', model_name = 'svm')
# Predict(model, model_name = 'svm', file_path = 'Emo-db/test/test.wav')

# model = Train(model_name = 'lstm', save_model_name = 'LSTM_LIBROSA', if_load = True)
# model = load_model(load_model_name = 'LSTM_LIBROSA', model_name = 'lstm')
# Predict(model, model_name = 'lstm', file_path = 'Emo-db/test/test.wav')

# model = Train(model_name = 'cnn', save_model_name = 'CNN_LIBROSA', if_load = True)
# model = load_model(load_model_name = 'CNN_LIBROSA', model_name = 'cnn')
# Predict(model, model_name = 'cnn', file_path = 'Emo-db/test/test.wav')

# model = Train(model_name = 'rnn', save_model_name = 'RNN_LIBROSA', if_load = True)
# model = load_model(load_model_name = 'RNN_LIBROSA', model_name = 'rnn')
# Predict(model, model_name = 'rnn', file_path = 'Emo-db/test/test.wav')

model = Train(model_name = 'lstm2', save_model_name = 'LSTM2', if_load = True)
model = load_model(load_model_name = 'LSTM2', model_name = 'lstm2')
Predict(model, model_name = 'lstm2', file_path = 'Emo-db/test/test.wav')

# model = Train(model_name = 'vgg', save_model_name = 'VGG', if_load = False)
# model = load_model(load_model_name = 'VGG', model_name = 'vgg')
# Predict(model, model_name = 'vgg', file_path = 'test_images/test.png')