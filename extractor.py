import os
# import sys
# import subprocess
import librosa
import librosa.display
import numpy as np
# from typing import Tuple
import pickle
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import config
from speechpy.feature import mfcc
import scipy.io.wavfile as wav
import Utils
# import scipy.misc
from skimage import io

def get_files_and_labels(path = config.DATA_PATH, extension = '.wav'):
	# get all file names 
	filepaths = os.listdir(path) 
	filenames = [path + '/' + filename for filename in filepaths if extension in filename]
	# create labels
	labels = []
	for i in range(len(filenames)):
		if(extension == '.wav'):
			labels.append(filenames[i][16])
		elif(extension == '.png'):
			labels.append(filenames[i][12])
	# print(filenames)
	return filenames, labels

def features(X, sample_rate):
	# short-time Fourier transform
	stft = np.abs(librosa.stft(X))
	
	# pitch(基频)
	# sample_rate: the number of sample extracted in 1 second. 
	# fmax, fmin: the max and min value of the pitch
	pitches, magnitudes = librosa.piptrack(X, sr = sample_rate, S = stft, fmin = 70, fmax = 400)
	pitch = []
	for i in range(magnitudes.shape[1]):
		index = magnitudes[:, 1].argmax()
		pitch.append(pitches[index, i])
	
	pitch_tuning_offset = librosa.pitch_tuning(pitches)
	pitchmean = np.mean(pitch)
	pitchstd = np.std(pitch)
	pitchmax = np.max(pitch)
	pitchmin = np.min(pitch)
	
	# Spectral Centroid(频谱质心)
	cent = librosa.feature.spectral_centroid(y = X, sr = sample_rate)
	cent = cent / np.sum(cent)
	centmean = np.mean(cent)
	centstd = np.std(cent)
	centmax = np.max(cent)
	
	# Spectral Flatness(谱平面)
	flatness = np.mean(librosa.feature.spectral_flatness(y = X))
	
	# MFCC(梅尔频率倒谱特征，系数为0)
	mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)
	mfccsstd = np.std(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)
	mfccmax = np.max(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)
	
	# chroma stft(色谱图)
	chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate).T, axis = 0)
	
	# Melspectrogram(梅尔频率)
	mel = np.mean(librosa.feature.melspectrogram(X, sr = sample_rate).T, axis = 0)
	
	# ottava contrast(ottava对比)
	contrast = np.mean(librosa.feature.spectral_contrast(S = stft, sr = sample_rate).T, axis = 0)
	
	# zero crossing rate(过零率)
	zerocr = np.mean(librosa.feature.zero_crossing_rate(X))
	
	# magphase: 复数图谱的幅度值和相位值
	S, phase = librosa.magphase(stft)
	Magnitudemean = np.mean(S)
	Magnitudestd = np.std(S)
	Magnitudemax = np.max(S)
	
	# rms(均方根能量)
	rms = librosa.feature.rms(S = S)[0]
	rmsmean = np.mean(rms)
	rmsstd = np.std(rms)
	rmsmax = np.max(rms)
	
	# all the features extracted
	ext_features = np.array([
		flatness, zerocr, Magnitudemean, Magnitudemax, centmean, centstd,
		centmax, Magnitudestd, pitchmean, pitchmax, pitchstd,
		pitch_tuning_offset, rmsmean, rmsmax, rmsstd
	])
	
	ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))
	
	return ext_features

'''def extract_features(filename, pad = False):
	print('opening: ' + filename)
	X, sample_rate = librosa.load(filename, sr = None)
	max_ = X.shape[0] / sample_rate
	# adding zeros to the end of the vector
	if pad:
		length = (max_ * sample_rate) - X.shape[0]
		X = np.pad(X, (0, int(length)), 'constant')
	return features(X, sample_rate)'''

def extract_features_2(filename, max_, pad = True):
	print('opening: ' + filename)
	X, sample_rate = librosa.load(filename, sr = None)
	# max_ = X.shape[0] / sample_rate
	# adding zeros to the end of the vector
	if pad:
		length = (max_ * sample_rate) - X.shape[0]
		X = np.pad(X, (0, int(length)), 'constant')
	return X

def get_max_min(filenames):
	min_, max_ = 100, 0
	
	for filename in filenames:
		sound_file, samplerate = librosa.load(filename, sr = None)
		t = sound_file.shape[0] / samplerate
		if t < min_:
			min_ = t
		if t > max_:
			max_ = t
	
	return max_, min_

def load_feature(feature_path, train: bool):
	features = pd.DataFrame(data = joblib.load(feature_path), columns = ['file_name', 'features', 'label'])
	
	print(features.shape)

	X = list(features['features'])
	Y = list(features['label'])
	
	if train:
		# standardization
		scaler = StandardScaler().fit(X)
		# save standardization model
		joblib.dump(scaler, config.MODEL_PATH + 'SCALER_LIBROSA.m')
		X = scaler.transform(X)
		
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)
		return x_train, x_test, y_train, y_test
	
	else:
		# standardization
		# load standardization model
		scaler = joblib.load(config.MODEL_PATH + 'SCALER_LIBROSA.m')
		X = scaler.transform(X)
		return X

def get_data(train: bool, data_path = config.DATA_PATH, feature_path = config.TRAIN_FEATURE_PATH_LIBROSA):
	if train:
		filenames, labels = get_files_and_labels(data_path)
		max_, min_ = get_max_min(filenames)
		
		mfcc_data = []
		for i, filename in enumerate(filenames):
			features = extract_features(filename, max_)
			label = labels[i]
			mfcc_data.append([filename, features, config.CLASS_LABELS.index(label)])
	
	else:
		features = extract_features(data_path)
		mfcc_data = [[data_path, features, -1]]
	
	cols = ['file_name', 'features', 'label']
	mfcc_pd = pd.DataFrame(data = mfcc_data, columns = cols)
	pickle.dump(mfcc_data, open(feature_path, 'wb'))
	
	return load_feature(feature_path, train = train)
	# load the file:
	# with open('Features/train_librosa_emodb.csv', 'rb') as f:
	# 	X = pickle.loads(f.read())

def get_feature_vector_from_mfcc(file_path, flatten: bool, mfcc_len = 39) -> np.ndarray:
	# make feature vector from MFCC for the given wav file.
    fs, signal = wav.read(file_path)
    s_len = len(signal)
    # pad the signals to have same size if lesser than required
    # else slice them
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    if flatten:
        # Flatten the data
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients

def get_data_cnn(data_path, flatten: bool = True, mfcc_len = 39):
	# extract data for training and testing
	filenames, labels = get_files_and_labels(data_path)
	data = []
	labels = [config.CLASS_LABELS.index(label) for label in labels]
	for filename in filenames:
		feature_vector = get_feature_vector_from_mfcc(file_path = filename, mfcc_len = mfcc_len, flatten = flatten)
		data.append(feature_vector)
	return np.array(data), np.array(labels)

def extract_data_cnn(flatten):
    data, labels = get_data_cnn(config.DATA_PATH, flatten = flatten)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), 7

def get_feature_vector_from_mfcc(file_path, flatten: bool, mfcc_len = 39) -> np.ndarray:
	mean_signal_length = 32000
	fs, signal = wav.read(file_path)
	s_len = len(signal)
	# pad the signals to have same size if lesser than required
	if s_len < mean_signal_length:
		pad_len = mean_signal_length - s_len
		pad_rem = pad_len % 2
		pad_len //= 2
		signal = np.pad(signal, (pad_len, pad_len + pad_rem),
						'constant', constant_values=0)
	else:
		pad_len = s_len - mean_signal_length
		pad_len //= 2
		signal = signal[pad_len:pad_len + mean_signal_length]
	mel_coefficients = mfcc(signal, fs, num_cepstral = mfcc_len)
	if flatten:
		# Flatten the data
		mel_coefficients = np.ravel(mel_coefficients)
	return mel_coefficients

def get_data_2(feature_path, train: bool, data_path = config.DATA_PATH):
	if train:
		filenames, labels = get_files_and_labels(data_path)
		max_, min_ = get_max_min(filenames)
		
		mfcc_data = []
		for i, filename in enumerate(filenames):
			feature = extract_features_2(filename, max_, True)
			label = labels[i]
			feature = np.array(feature)
			mfcc_data.append([filename, feature, config.CLASS_LABELS.index(label)])

	else:
		T, L = get_files_and_labels(config.DATA_PATH)
		max_, min_ = get_max_min(T)

		feature = extract_features_2(data_path, max_, True)
		feature = np.array(feature)
		mfcc_data = [[data_path, feature, -1]]
	
	cols = ['file_name', 'features', 'label']
	mfcc_pd = pd.DataFrame(data = mfcc_data, columns = cols)
	pickle.dump(mfcc_data, open(feature_path, 'wb'))
	
	return load_feature(feature_path, train = train)
	# load the file:
	# with open('Features/train_librosa_emodb.csv', 'rb') as f:
	# 	X = pickle.loads(f.read())

'''def get_images(filepath, save_path):
	filenames, labels = get_files_and_labels(config.DATA_PATH)
	for filename in filenames:
		plt = Utils.Waveform_1(filename)
		savepath = save_path + '/' + str(filename[11:18]) + '.png'
		plt.savefig(savepath)
		plt.close()

def imread(path):
	# return scipy.misc.imread(path).astype(np.float)
	return io.imread(path, as_gray = True)

def load_images(feature_path, train: bool):
	images, labels = get_files_and_labels(path = config.IMAGES_PATH, extension = '.png')
	
	X = []
	for image in images:
		# img = list(imread(image))
		print(image)
		X.append(imread(image))
	
	X = np.array(X)

	Y = [config.CLASS_LABELS.index(label) for label in labels]
	
	if train:
		
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)
		return x_train, x_test, y_train, y_test
	
	else:

		return X'''

# get_data(data_path = 'Emo-db/wav', feature_path = 'Features/train_librosa_emodb.csv', train = True)
# get_data(data_path = 'Emo-db/test/test.wav', feature_path = 'Features/test_librosa_emodb.csv', train = False)

# get_data_2(data_path = 'Emo-db/wav', feature_path = 'Features/train_new.csv', train = True)
# get_data_2(data_path = 'Emo-db/test/test.wav', feature_path = 'Features/test_new.csv', train = False)

# get_images(config.DATA_PATH, config.IMAGES_PATH)