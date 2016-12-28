from os import listdir
import json
import random
import time
import cv2
import pickle
import numpy as np

from constants import *


class Dataset():
    def __init__(self, data_folder, file_name=None):
        # load dataset from pickle files
        self._load_data(data_folder)

        self.train_idx = []
        self.val_idx = []

        if not file_name:
            # create new training and validation set
            self._create_new_train_val_set()
        else:
            # read dataset from files
            self._read_train_val_set(file_name)

        self.train_idx = np.array(self.train_idx, dtype='int')
        self.val_idx = np.array(self.val_idx, dtype='int')

    def get_train_dataset(self):
        return self._get_dataset(self.train_idx)

    def get_val_dataset(self):
        return self._get_dataset(self.val_idx)

    def _get_dataset(self, idx_list):
        m = idx_list.shape[0]
        X = np.zeros((m, HEIGHT, WIDTH, 1), dtype='float32')
        Y = np.zeros((m, 2), dtype='int')
        for i in range(m):
            X[i] = self.X[idx_list[i]]
            Y[i] = self.Y[idx_list[i]]

        return X, Y

    def _load_data(self, data_folder):
        self.X = None
        self.Y = None
        first = True
        for file in listdir(data_folder):
            with open(data_folder + file, "rb") as f:
                if first:
                    first = False
                    self.X = pickle.load(f)
                    self.Y = pickle.load(f)
                else:
                    self.X = np.concatenate((self.X, pickle.load(f)))
                    self.Y = np.concatenate((self.Y, pickle.load(f)))

    def _create_new_train_val_set(self, ratio=0.8):
        # select training set and validation set from data randomly
        for i in range(self.X.shape[0]):
            r = random.random()
            if r < ratio:
                self.train_idx.append(i)
            else:
                self.val_idx.append(i)

        random.shuffle(self.train_idx)
        random.shuffle(self.val_idx)

        # save their indexes to files
        with open("train_val_set_{}".format(self.X.shape[0]), "w") as f:
            f.write(json.dumps({
                "train_idx": self.train_idx,
                "val_idx": self.val_idx
            }))

    def _read_train_val_set(self, file_name):
        with open(file_name, "r") as f:
            obj = json.loads(f.read())
        self.train_idx = obj["train_idx"]
        self.val_idx = obj["val_idx"]


if __name__ == "__main__":  # process raw data
    # read some random file names in folder
    # number = 75k each label
    for i in range(2):
        all_names = np.array(["{}/{}".format(i, j) for j in random.sample(listdir(DATASET_FOLDER + str(i)), 75000)])
        m = all_names.shape[0]

        X = np.zeros((m, HEIGHT * WIDTH))
        Y = np.zeros((m, 2), dtype='int')

        for k in range(m):
            img_src = DATASET_FOLDER + all_names[k]
            # load image as grayscale
            img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
            # flat image
            img_flat = img.flatten() / 255.0  # normalize from [0, 255] to [0, 1]
            X[k] = img_flat
            Y[k, i] = 1

        X = X.reshape(m, HEIGHT, WIDTH, 1).astype('float32')

        with open("{}{}".format(PICKLE_DATASET, i), "wb") as f:
            pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(Y, f, protocol=pickle.HIGHEST_PROTOCOL)
