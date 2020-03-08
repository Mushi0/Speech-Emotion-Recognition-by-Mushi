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

def get_files_and_labels(path = 'Emo-db/wav', extension='.wav'):
	# get all file names 
	filepaths = os.listdir(path) 
	filenames = [path + '/' + filename for filename in filepaths if extension in filename]
	# create labels
	labels = []
	for i in range(len(filenames)):
		labels.append(filenames[i][16])
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

def extract_features(filename, pad = False):
	print('opening: ' + filename)
	X, sample_rate = librosa.load(filename, sr = None)
	max_ = X.shape[0] / sample_rate
	# adding zeros to the end of the vector
	if pad:
		length = (max_ * sample_rate) - X.shape[0]
		X = np.pad(X, (0, int(length)), 'constant')
	return features(X, sample_rate)

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

def get_data(train: bool, data_path = 'Emo-db/wav', feature_path = 'Features/train_librosa_emodb.csv'):
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

# get_data(data_path = 'Emo-db/wav', feature_path = 'Features/train_librosa_emodb.csv', train = True)
# get_data(data_path = 'Emo-db/test/test.wav', feature_path = 'Features/test_librosa_emodb.csv', train = False)
