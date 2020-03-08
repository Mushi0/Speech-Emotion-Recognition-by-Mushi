import numpy as np
from keras.utils import np_utils
import os

from MLP_SVM import SVM_Model, MLP_Model
# from DNN_Model import LSTM_Model

from Utils import load_model, Radar

import extractor as ex
from config import config

# the traing fuction
# model_name: the name of the model
# save_model_name: the name of the model to be saved with
# if_load: if the features has been loaded
def Train(model_name, save_model_name, if_load = True):
	if(if_load == True):
		x_train, x_test, y_train, y_test = ex.load_feature(feature_path = config.TRAIN_FEATURE_PATH_LIBROSA, train = True)
	else:
		x_train, x_test, y_train, y_test = ex.get_data(config.DATA_PATH, config.TRAIN_FEATURE_PATH_LIBROSA, train = True)
	
	# build the model
	if(model_name == 'svm'):
		model = SVM_Model()
	elif(model_name == 'mlp'):
		model = MLP_Model()
	elif(model_name == 'lstm'):
		y_train = np_utils.to_categorical(y_train)
		y_val = np_utils.to_categorical(y_test)
		
		model = LSTM_Model(input_shape = x_train.shape[1], num_classes = len(config.CLASS_LABELS))
		
		x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
		x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
	
	print('-------------------------------- Start --------------------------------')
	if(model_name == 'svm' or model_name == 'mlp'):
		model.train(x_train, y_train)
	elif(model_name == 'lstm'):
		model.train(x_train, y_train, x_test, y_val, n_epochs = config.epochs)
	
	model.evaluate(x_test, y_test)
	model.save_model(save_model_name)
	print('---------------------------------- End ----------------------------------')
	
	return model


'''
Predict(): 预测音频情感
输入:
	model: 已加载或训练的模型
	model_name: 模型名称
	file_path: 要预测的文件路径
	feature_method: 提取特征的方法（'o': Opensmile / 'l': librosa）
输出：
	预测结果和置信概率
'''
# prediction
# model: the model that has been trained
# model_name: the name of the model
# file_path: the path of files to be predicted
def Predict(model, model_name, file_path):
	# file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
	test_feature = ex.get_data(False, file_path, config.PREDICT_FEATURE_PATH_LIBROSA)
	
	if(model_name == 'lstm'):
		test_feature = np.reshape(test_feature, (test_feature.shape[0], 1, test_feature.shape[1]))
	
	result = model.predict(test_feature)
	
	if(model_name == 'lstm'):
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

model = Train(model_name = 'lstm', save_model_name = 'LSTM_LIBROSA', if_load = True)
model = load_model(load_model_name = 'LSTM_LIBROSA', model_name = 'lstm')
Predict(model, model_name = 'lstm', file_path = 'Emo-db/test/test.wav')

# model = Train(model_name = 'lstm', save_model_name = 'LSTM_OPENSMILE_1', if_load = True, feature_method = 'o')
# model = load_model(load_model_name = 'LSTM_OPENSMILE', model_name = 'lstm')
# Predict(model, model_name = 'lstm', file_path = 'Test/neutral.wav', feature_method = 'o')
