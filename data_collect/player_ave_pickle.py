"""
This file collects player overall stats, and take average for teammates,
then take difference between red and blue teams
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
    # 19: overall statistics
    num_feat = 19
    output_file_name = 'lol_player_ave'

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

            # p_feature is player overall infor
            player_stats = mypymongo.find_match_history_in_match(account_id, match_create_time)
            p_feature = ave_stats(player_stats, ['win', 'assists', 'deaths', 'kills', 'champLevel',
                                                  'goldEarned', 'goldSpent', 'killingSprees',
                                                  'magicDamageDealtToChampions', 'magicalDamageTaken',
                                                  'physicalDamageDealtToChampions', 'physicalDamageTaken',
                                                  'totalHeal', 'timeCCingOthers', 'totalTimeCrowdControlDealt',
                                                  'wardsPlaced', 'wardsKilled', 'totalMinionsKilled'])
            # add number of matches overall
            feature = numpy.hstack((p_feature, len(player_stats)))

            if side == 'blue':
                feature = - feature

            x += feature

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
