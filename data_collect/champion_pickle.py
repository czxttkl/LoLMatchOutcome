"""
This file collects data with champion vector
"""
import numpy
from data_collect.mypymongo import MyPyMongo
import pprint
import pickle
from data_util import ave_stats, sum_stats, if_red_team_win
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    mypymongo = MyPyMongo()

    with open('../input/lol_basic.pickle', 'rb') as f:
        champion_id2idx_dict, M = pickle.load(f)
    X, Y = [], []

    # 19: in-game statistics
    # 2: overall win rate, overall num matches
    # M: one hot encoding of champions
    num_feat = M
    output_file_name = 'lol_champion'

    total_match = 0
    for match in mypymongo.db.match_seed.find({'all_participants_matches_crawled': True}, no_cursor_timeout=True):
        match_create_time = match['gameCreation']
        match_id = match['id']
        red_team_win = if_red_team_win(match)
        Y.append(red_team_win)
        x = numpy.zeros(num_feat)

        for participant in match['participants']:
            account_id = participant['accountId']
            champion_id = participant['championId']
            champion_idx = champion_id2idx_dict[champion_id]
            side = participant['side']

            c_feature = numpy.zeros(M)
            c_feature[champion_idx] = 1
            feature = c_feature

            if side == 'blue':
                feature = - feature

            x += feature

        assert numpy.sum(x == -1) == 5
        assert numpy.sum(x == 1) == 5
        X.append(x)

        total_match += 1
        print('match', total_match)

    X = numpy.array(X)
    Y = numpy.array(Y)
    shuffle_idx = numpy.arange(len(Y))
    numpy.random.shuffle(shuffle_idx)
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]

    with open('../input/{}.pickle'.format(output_file_name), 'wb') as f:
        pickle.dump((X, Y), f)
