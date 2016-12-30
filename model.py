import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from dataset import Dataset
import model_handler
from constants import *
from utils import Timer


def build_model():
    print("Building the model...")
    model = Sequential()
    model.add(Convolution2D(
        nb_filter=15,
        nb_row=6,
        nb_col=6,
        subsample=(2, 2),
        border_mode='valid',
        input_shape=(HEIGHT, WIDTH, 1),
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

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
    n_epochs = 30
    # the training may be slow depending on your computer
    model.fit(X_train,
              Y_train,
              batch_size=batch_size,
              nb_epoch=n_epochs,
              validation_data=(X_val, Y_val))

if __name__ == "__main__":
    # fix random seed for reproducibility
    seed = 13
    np.random.seed(seed)

    # deal with dataset
    timer = Timer()
    timer.start("Loading data")
    d = Dataset(PICKLE_DATASET + "12345_100k/", "train_val_set_100000")
    X_train, Y_train = d.get_train_dataset()
    X_val, Y_val = d.get_val_dataset()
    timer.stop()

    m = build_model()

    train_model(m)

    name = input("Model's name or 'n': ")
    if name != 'n':
        model_handler.save_model(m, "trained_models/" + name)
