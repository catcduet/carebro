import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from dataset import Dataset


HEIGHT = 18
WIDTH = 273

# fix random seed for reproducibility
seed = 13
np.random.seed(seed)

# deal with dataset
d = dataset.Dataset()
X_train, Y_train = d.get_train_dataset()
X_val, Y_val = d.get_val_dataset()


def build_model():
	print("Building the model...")
	model = Sequential()
	model.add(Convolution2D(
		nb_filter=32,
		nb_row=18,
		nb_col=18,
		subsample=(6, 6),
		border_mode='valid',
		dim_ordering='th',
		input_shape=(1, height, width),
		activation='relu'
	))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3)))

	model.add(Convolution2D(
		nb_filter=64,
		nb_row=10,
		nb_col=10,
		subsample=(1, 1),
		activation='relu'
	))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3)))

	model.add(Dense(2048, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(2048, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(WIDTH, activation='softmax'))

	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)

	return model


def train_model(model):
	print("Training the model...")
	# how many examples to look at during each training iteration
	batch_size = 128
	# how many times to run through the full set of examples
	n_epochs = 10
	# the training may be slow depending on your computer
	model.fit(X_train,
			Y_train,
			batch_size=batch_size,
			nb_epoch=n_epochs,
			validation_data=(X_val, Y_val))
