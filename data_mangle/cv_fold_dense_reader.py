""" Data reader for cross validation.
This reader only works for HotS, HoN and Dota datatsets.
This reader returns dense feature vectors
"""
from sklearn.model_selection import ShuffleSplit
import pickle
import numpy
import time


class CVFoldDenseReader(object):
    def __init__(self, data_path, folds, seed=None):
        """
        Read data files and then split data into K folds.
        """
        self.data_src = data_path.split('/')[-1].split('.')[0]
        self.read_data_from_file(data_path, folds, seed)

    def read_data_from_file(self, data_path, folds, seed):
        """
        Data format:
        """
        with open(data_path, 'rb') as f:
            self.X, self.Y = pickle.load(f)
            self.Z = len(self.Y)
            self.K = len(self.X[0])

        print("CVFoldReader. Z: {0}, K: {1}".format(self.Z, self.K))

        self.data_path = data_path
        self.folds = folds

        # use 90% as train data and 10% test data
        ss = ShuffleSplit(n_splits=folds, test_size=0.1, train_size=0.9, random_state=seed)
        self.train_split_idx = {}
        self.test_split_idx = {}

        i = 0
        for train_idx, test_idx in ss.split(numpy.arange(self.Z)):
            self.train_split_idx[i], self.test_split_idx[i] = train_idx, test_idx
            i += 1

    def read_train_test_fold(self, i):
        """
        Read i-th fold of splitted train/test data
        """
        train_idx, test_idx = self.train_split_idx[i], self.test_split_idx[i]

        X_train = self.X[train_idx]
        Y_train = self.Y[train_idx]
        X_test = self.X[test_idx]
        Y_test = self.Y[test_idx]

        return X_train, Y_train, X_test, Y_test

