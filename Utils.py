from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.externals import joblib
import librosa
import librosa.display
import scipy.io.wavfile as wav
# import pyaudio
import wave

from config import config

def load_model(load_model_name, model_name):
	if(model_name == 'svm' or model_name == 'mlp'):
		model_path = 'Models/' + load_model_name + '.m'
		model = joblib.load(model_path)
	
	elif(model_name == 'lstm' or model_name == 'cnn'):
		# load json
		model_path = 'Models/' + load_model_name + '.h5'
		model_json_path = 'Models/' + load_model_name + '.json'
		
		json_file = open(model_json_path, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		
		# load weights
		model.load_weights(model_path)

	return model

# plot the loss and accurency
# train: the loss and accuracy_score of training set
# val: the loss and accuracy_score of test set
def plotCurve(train, val, title, y_label):
	plt.plot(train)
	plt.plot(val)
	plt.title(title)
	plt.ylabel(y_label)
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

'''def playAudio(file_path = 'Emo-db/wav'):
	p = pyaudio.PyAudio()
	f = wave.open(file_path, 'rb')
	stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
					channels = f.getnchannels(),
					rate = f.getframerate(),
					output = True)
	data = f.readframes(f.getparams()[3])
	stream.write(data)
	stream.stop_stream()
	stream.close()
	f.close()
'''

# draw radar chart of confidence ratio
# data_prob(numpy.ndarray): 概率数组
def Radar(data_prob):
	angles = np.linspace(0, 2 * np.pi, len(config.CLASS_LABELS), endpoint = False)
	data = np.concatenate((data_prob, [data_prob[0]])) 
	angles = np.concatenate((angles, [angles[0]]))
	
	fig = plt.figure()
	
	ax = fig.add_subplot(111, polar = True)
	ax.plot(angles, data, 'bo-', linewidth=2)
	ax.fill(angles, data, facecolor='r', alpha=0.25)
	ax.set_thetagrids(angles * 180 / np.pi, config.CLASS_LABELS)
	ax.set_title("Emotion Recognition", va = 'bottom')
	
	ax.set_rlim(0, 1)
	
	ax.grid(True)
	# plt.ion()
	plt.show()
	# plt.pause(4)
	# plt.close()

# draw the waveform
def Waveform(file_path):
	data, sampling_rate = librosa.load(file_path)
	plt.figure(figsize=(15, 5))
	librosa.display.waveplot(data, sr = sampling_rate)
	plt.show()

# draw the spectrogram
def Spectrogram(file_path):
	sr, x = wav.read(file_path)
	
	# step: 10ms, window: 30ms
	nstep = int(sr * 0.01)
	nwin  = int(sr * 0.03)
	nfft = nwin
	window = np.hamming(nwin)
	
	nn = range(nwin, len(x), nstep)
	X = np.zeros( (len(nn), nfft//2) )
	
	for i,n in enumerate(nn):
		xseg = x[n-nwin:n]
		z = np.fft.fft(window * xseg, nfft)
		X[i,:] = np.log(np.abs(z[:nfft//2]))
	
	plt.imshow(X.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto')
	plt.show()