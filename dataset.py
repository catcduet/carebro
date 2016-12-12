from os import listdir
from os.path import isfile, join
import json
import random
import time
import cv2
import numpy as np


DATASET_FOLDER = "XXX/"
HEIGHT = 18
WIDTH = 273

class Dataset():
    def __init__(self, file_name):      
        self.all_names = np.array(listdir(DATASET_FOLDER))  # read all file names in folder
        self.all_labels = []  # [0, 272]
        self.train_idx = []
        self.val_idx = []

        # extract all labels from names
        for i in range(self.all_names.shape[0]):
            name = self.all_names[i].split(".")
            self.all_labels.append(int(name[x]))
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
        return self._get_dataset(self.train_idx)

    def get_val_dataset(self):
        return self._get_dataset(self.val_idx)

    def _get_dataset(self, idx_list):
        m = idx_list.shape[0]
        X = np.zeros((m, HEIGHT * WIDTH))
        Y = np.zeros((m, WIDTH), dtype='int')
        I = np.eye(WIDTH)

        for i in range(m):
            img_idx = idx_list[i]
            img_src = DATASET_FOLDER + self.all_names[img_idx] + ".png"
            # load image as grayscale
            img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
            # flat image
            img_flat = img.flatten() / 255.0  # normalize from [0, 255] to [0, 1]
            X[i] = img_flat
            Y[i, :] = I[self.all_labels[img_idx], :]
            
        X = X.reshape(m, 1, HEIGHT, WIDTH).astype('float32')

        return X, Y

    def _create_new_train_val_set(self, ratio=0.8):
        # select training set and validation set from data randomly
        for i in range(self.num_data):
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

    def _read_train_val_set(self, file_name):
        with open(file_name, "r") as f:
            obj = json.loads(f.read())
        self.train_idx = obj["train_idx"]
        self.val_idx = obj["val_idx"]
