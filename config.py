# a configuration of parameters

class config:
	# path of database
	DATA_PATH = 'Emo-db/wav'
	# labels
	CLASS_LABELS = ('W', 'L', 'E', 'A', 'F', 'T', 'N')
	
	# epoch value in LSTM
	epochs = 20
	
	# path of features
	FEATURE_PATH = 'Features/'
	# # path of taining features by opensmile
	#  TRAIN_FEATURE_PATH_OPENSMILE = FEATURE_PATH + 'train_opensmile_casia.csv'
	# # path of test features by opensmile
	# PREDICT_FEATURE_PATH_OPENSMILE = FEATURE_PATH + 'test_opensmile_casia.csv'
	# path of taining features by librosa
	TRAIN_FEATURE_PATH_LIBROSA = FEATURE_PATH + 'train_librosa_emodb.csv'
	# path of test features by librosa
	PREDICT_FEATURE_PATH_LIBROSA = FEATURE_PATH + 'test_librosa_emodb.csv'

	TRAIN_FEATURE_PATH_NEW = FEATURE_PATH + 'train_new.csv'

	PREDICT_FEATURE_PATH_NEW = FEATURE_PATH + 'test_new.csv'
	
	IMAGES_PATH = 'images'

	IMAGES_PATH_TEST = 'test_images'

	# path of models
	MODEL_PATH = 'Models/'
