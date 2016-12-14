from os import listdir
from os.path import isfile, join
import json
import random
import time
import cv2
import pickle
import numpy as np
from constants import *


class Dataset():
    def __init__(self, file_name=None):
        # read all file names in folder
        self.all_names = []
        for i in range(WIDTH + 1):
            self.all_names.extend(["{}/{}".format(i, j) for j in listdir(DATASET_FOLDER + str(i))])
        self.all_names = np.array(self.all_names)
        self.all_labels = []  # [0, 272]
        self.train_idx = []
        self.val_idx = []

        # extract all labels from names
        for i in range(self.all_names.shape[0]):
            name = self.all_names[i].split("/")
            self.all_labels.append(int(name[0]))
        self.all_labels = np.array(self.all_labels, dtype='int')

        if not file_name:
            # create new training and validation set
            self._create_new_train_val_set()
        else:
            # read dataset from files
            self._read_train_val_set(file_name)

        self.train_idx = np.array(self.train_idx, dtype='int')
        self.val_idx = np.array(self.val_idx, dtype='int')

    def get_train_dataset(self):
        return self.load_pickle_data(TRAIN_DATASET_FILE)

    def get_val_dataset(self):
        return self.load_pickle_data(VAL_DATASET_FILE)

    def _create_new_train_val_set(self, ratio=0.8):
        # select training set and validation set from data randomly
        for i in range(self.all_names.shape[0]):
            r = random.random()
            if r < ratio:
                self.train_idx.append(i)
            else:
                self.val_idx.append(i)

        # save their indexes to files
        with open("train_val_set_{}".format(int(time.time())), "w") as f:
            f.write(json.dumps({
                "train_idx": self.train_idx,
                "val_idx": self.val_idx
            }))

    def load_raw_data(self, idx_list):
        m = idx_list.shape[0]
        X = np.zeros((m, HEIGHT * WIDTH))
        Y = np.zeros((m, WIDTH+1), dtype='int')
        I = np.eye(WIDTH+1)

        for i in range(m):
            img_idx = idx_list[i]
            img_src = DATASET_FOLDER + self.all_names[img_idx]
            # load image as grayscale
            img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
            # flat image
            img_flat = img.flatten() / 255.0  # normalize from [0, 255] to [0, 1]
            X[i] = img_flat
            Y[i, :] = I[self.all_labels[img_idx], :]

        X = X.reshape(m, 1, HEIGHT, WIDTH).astype('float32')

        return X, Y

    def dump_data(self, images, labels, filename):
        with open(filename, "wb") as f:
            pickle.dump(images, f)
            pickle.dump(labels, f)

    def load_pickle_data(self, filename):
        with open(filename, "rb") as f:
            images = pickle.load(f)
            labels = pickle.load(f)

        return images, labels

    def _read_train_val_set(self, file_name):
        with open(file_name, "r") as f:
            obj = json.loads(f.read())
        self.train_idx = obj["train_idx"]
        self.val_idx = obj["val_idx"]

if __name__ == "__main__":
    d = Dataset("train_val_set_1481707248")
    X_train, Y_train = d.load_raw_data(d.train_idx)
    X_val, Y_val = d.load_raw_data(d.val_idx)
    d.dump_data(X_train, Y_train, TRAIN_DATASET_FILE)
    d.dump_data(X_val, Y_val, VAL_DATASET_FILE)