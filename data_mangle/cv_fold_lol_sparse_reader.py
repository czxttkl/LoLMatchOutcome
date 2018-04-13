""" Data reader for cross validation.
This reader only works for League of Legend data.
This reader returns sparse feature vectors
"""
from sklearn.model_selection import ShuffleSplit
import pickle
import numpy
import time
from scipy.sparse import csr_matrix


class CVFoldLoLSparseReader(object):
    def __init__(self, data_path, folds, feature_config, seed=None):
        """
        Read data files and then split data into K folds.
        """
        self.data_src = data_path.split('/')[-1].split('.')[0]
        self.feature_config = feature_config
        self.read_data_from_file(data_path, folds, seed)

    def read_data_from_file(self, data_path, folds, seed):
        """
        Data format:

        M_o (n_matches): 1D vector of outcomes, in which 0 denotes the blue team wins and 1 denotes the red team wins

        M_r_C (n_matches, 5): avatar idxs in the red team for each match

        M_b_C (n_matches, 5): avatar idxs in the blue team for each match

        M_r_P (n_matches, 5): summoner idxs in the red team for each match

        M_b_P (n_matches, 5): summoner idxs in the blue team for each match

        M: the number of avatars in the dataset.
        """
        with open(data_path, 'rb') as f:
            self.M_o, self.M_r_C, self.M_b_C, self.M_r_P, self.M_b_P, \
            self.match_id2idx_dict, self.summoner_id2idx_dict, self.champion_id2idx_dict, self.Z, self.N, self.M = \
                pickle.load(f)

        print("CVFoldReader. Z: {}, N: {}, M: {}".format(self.Z, self.N, self.M))

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

        M_o_train = self.M_o[train_idx]
        M_r_C_train = self.M_r_C[train_idx]
        M_b_C_train = self.M_b_C[train_idx]
        M_r_P_train = self.M_r_P[train_idx]
        M_b_P_train = self.M_b_P[train_idx]
        M_o_test = self.M_o[test_idx]
        M_r_C_test = self.M_r_C[test_idx]
        M_b_C_test = self.M_b_C[test_idx]
        M_r_P_test = self.M_r_P[test_idx]
        M_b_P_test = self.M_b_P[test_idx]

        train_feature = self.to_feature_matrix(M_r_C_train, M_b_C_train, M_r_P_train, M_b_P_train)
        test_feature = self.to_feature_matrix(M_r_C_test, M_b_C_test, M_r_P_test, M_b_P_test)

        return train_feature, M_o_train, test_feature, M_o_test

    def to_feature_matrix(self, M_r_C, M_b_C, M_r_P, M_b_P):
        assert M_r_C.shape[0] == M_b_C.shape[0] == M_r_P.shape[0] == M_b_P.shape[0]
        t1 = time.time()
        Z = M_r_C.shape[0]

        if self.feature_config == 'champion_two_teams':
            row = numpy.repeat(numpy.arange(Z), 10)
            col = numpy.hstack((M_r_C, M_b_C + self.M)).flatten()  # blue team champion idxs should shift M
            assert len(row) == len(col)
            vals = numpy.ones((len(row),))
            data = csr_matrix((vals, (row, col)), shape=(Z, 2 * self.M))
        elif self.feature_config == 'summoner_two_teams':
            row = numpy.repeat(numpy.arange(Z), 10)
            col = numpy.hstack((M_r_P, M_b_P + self.N)).flatten()   # blue team summoner idxs should shift N
            assert len(row) == len(col)
            vals = numpy.ones((len(row),))
            data = csr_matrix((vals, (row, col)), shape=(Z, 2 * self.N))
        elif self.feature_config == 'champion_summoner_two_teams':
            row = numpy.repeat(numpy.arange(Z), 20)
            col = numpy.hstack((M_r_C, M_b_C + self.M, M_r_P + 2 * self.M, M_b_P + 2 * self.M + self.N)).flatten()
            assert len(row) == len(col)
            vals = numpy.ones((len(row),))
            data = csr_matrix((vals, (row, col)), shape=(Z, 2 * (self.M + self.N)))
        elif self.feature_config == 'champion_one_team':
            row = numpy.repeat(numpy.arange(Z), 10)
            col = numpy.hstack((M_r_C, M_b_C)).flatten()  # blue team champion idxs should shift M
            assert len(row) == len(col)
            vals = numpy.tile([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], Z)
            data = csr_matrix((vals, (row, col)), shape=(Z, self.M))
        elif self.feature_config == 'summoner_one_team':
            row = numpy.repeat(numpy.arange(Z), 10)
            col = numpy.hstack((M_r_P, M_b_P)).flatten()  # blue team champion idxs should shift M
            assert len(row) == len(col)
            vals = numpy.tile([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], Z)
            data = csr_matrix((vals, (row, col)), shape=(Z, self.N))
        elif self.feature_config == 'champion_summoner_one_team':
            row = numpy.repeat(numpy.arange(Z), 20)
            col = numpy.hstack((M_r_C, M_b_C, M_r_P + self.M, M_b_P + self.M)).flatten()
            assert len(row) == len(col)
            vals = numpy.tile([1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
                               1, 1, 1, 1, 1, -1, -1, -1, -1, -1], Z)
            data = csr_matrix((vals, (row, col)), shape=(Z, self.M + self.N))
        else:
            raise NotImplementedError

        print("finish feature matrix conversion. time:", time.time() - t1)
        return data

    def print_feature_config(self):
        # only champion information
        if self.feature_config == 'champion_two_teams':
            return "champion_two_teams_sparse"
        # only summoner information
        elif self.feature_config == 'summoner_two_teams':
            return "summoner_two_teams_sparse"
        # summoner + champion information
        elif self.feature_config == 'champion_summoner_two_teams':
            return "champion_summoner_two_teams_sparse"
        # champion one team
        if self.feature_config == 'champion_one_team':
            return "champion_one_team_sparse"
        # summoner one team
        elif self.feature_config == 'summoner_one_team':
            return "summoner_one_team_sparse"
        # champion summoner one team
        elif self.feature_config == 'champion_summoner_one_team':
            return "champion_summoner_one_team_sparse"
        else:
            raise NotImplementedError
