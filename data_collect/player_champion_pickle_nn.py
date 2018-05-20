"""
This file collects data with both player and champion information
(player champion specific stat + player overall stat) for each champion
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
    num_feat_per_champion = 21
    output_file_name = 'lol_player_champion_nn'

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
            pc_feature = ave_stats(player_stats, ['win', 'assists', 'deaths', 'kills', 'champLevel',
                                                  'goldEarned', 'goldSpent', 'killingSprees',
                                                  'magicDamageDealtToChampions', 'magicalDamageTaken',
                                                  'physicalDamageDealtToChampions', 'physicalDamageTaken',
                                                  'totalHeal', 'timeCCingOthers', 'totalTimeCrowdControlDealt',
                                                  'wardsPlaced', 'wardsKilled', 'totalMinionsKilled'])
            # add number of matches for the specific champion
            pc_feature = numpy.hstack((pc_feature, len(player_stats)))

            # p feature is player overall info
            player_stats = mypymongo.find_match_history_in_match(account_id, match_create_time)
            p_feature = numpy.hstack((ave_stats(player_stats, ['win']), len(player_stats)))

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

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    with open('../input/{}.pickle'.format(output_file_name), 'wb') as f:
        pickle.dump((X, Y), f)
