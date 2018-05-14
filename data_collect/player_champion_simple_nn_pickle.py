"""
This file collects data with both player and champion information:
for each champion block, number of wins and number of matches for overall and champion specific
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
    # 2: champion specific num win, champion specific num matches
    # 2: overall num win, overall num matches
    num_feat_per_champion = 4
    output_file_name = 'lol_player_champion_simple_nn'

    total_match = 0
    for match in mypymongo.db.match_seed.find({'all_participants_matches_crawled': True}, no_cursor_timeout=True):
        match_create_time = match['gameCreation']
        match_id = match['id']
        red_team_win = if_red_team_win(match)
        Y.append(red_team_win)
        x = numpy.zeros(M * num_feat_per_champion)

        for participant in match['participants']:
            account_id = participant['accountId']
            champion_id = participant['championId']
            champion_idx = champion_id2idx_dict[champion_id]
            side = participant['side']

            # pc_feature is player-champion specific
            player_stats = mypymongo.find_match_history_in_match(account_id, match_create_time, champion_id=champion_id)
            pc_feature = sum_stats(player_stats, ['win', 'presence'])

            # p feature is player overall info
            player_stats = mypymongo.find_match_history_in_match(account_id, match_create_time)
            p_feature = sum_stats(player_stats, ['win', 'presence'])

            feature = numpy.hstack((pc_feature, p_feature))

            if side == 'blue':
                feature = - feature

            start_idx = champion_idx * num_feat_per_champion
            x[start_idx : start_idx + num_feat_per_champion] = feature

        X.append(x)

        total_match += 1
        print('match', total_match)

    X = numpy.array(X)
    Y = numpy.array(Y)
    shuffle_idx = numpy.arange(len(Y))
    numpy.random.shuffle(shuffle_idx)
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]

    # no need to scale because all feature are numbers of matches
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    with open('../input/{}.pickle'.format(output_file_name), 'wb') as f:
        pickle.dump((X, Y), f)
