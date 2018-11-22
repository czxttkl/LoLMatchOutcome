"""
Use this file to generate player history features
Output a dictionary (plus some auxiliary information) whose key is player index, and value is a matrix
Each row of the matrix is a match's feature vector.
A match's feature vector contains time match time information, match statistics,
one-hot encodings of role, lane, and champion used

Running this file requires connection to mongodb
cd mongodb-linux-x86_64-ubuntu1604-3.6.2/
bin/mongod --dbpath <data_path>

See data/README.md for how to import mongodb data
"""
import pickle
import time
from mypymongo import MyPyMongo
import numpy
import os


# features we will definitely not use
useless_features = {
    'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'participantId',
    'perkPrimaryStyle', 'perkSubStyle', 'presence',
    'playerScore0', 'playerScore1', 'playerScore2', 'playerScore3', 'playerScore4',
    'playerScore5', 'playerScore6', 'playerScore7', 'playerScore8', 'playerScore9',
}

# useful features excluding role and lane.
# Role and lane features will be represented in another feature sets.
# commented features have most values of zeroes, so we do not use them for the time being
useful_features = [
    'assists',
    'champLevel',
    # 'combatPlayerScore',
    'damageDealtToObjectives',
    'damageDealtToTurrets',
    'damageSelfMitigated',
    'deaths',
    'doubleKills',
    'duration',
    # 'firstBloodAssist',
    'firstBloodKill',
    'firstTowerAssist',
    'firstTowerKill',
    'goldEarned',
    'goldSpent',
    'inhibitorKills',
    'killingSprees',
    'kills',
    'largestCriticalStrike',
    'largestKillingSpree',
    'largestMultiKill',
    'longestTimeSpentLiving',
    'magicDamageDealt',
    'magicDamageDealtToChampions',
    'magicalDamageTaken',
    'neutralMinionsKilled',
    'neutralMinionsKilledEnemyJungle',
    'neutralMinionsKilledTeamJungle',
    'pentaKills',
    'physicalDamageDealt',
    'physicalDamageDealtToChampions',
    'physicalDamageTaken',
    'quadraKills',
    # 'sightWardsBoughtInGame',
    'timeCCingOthers',
    'totalDamageDealt',
    'totalDamageDealtToChampions',
    'totalDamageTaken',
    'totalHeal',
    'totalMinionsKilled',
    # 'totalPlayerScore',
    # 'totalScoreRank',
    'totalTimeCrowdControlDealt',
    'totalUnitsHealed',
    'tripleKills',
    'trueDamageDealt',
    'trueDamageDealtToChampions',
    'trueDamageTaken',
    'turretKills',
    'unrealKills',
    'visionScore',
    'visionWardsBoughtInGame',
    'wardsKilled',
    'wardsPlaced',
    'win'
]

role_dict = {'SOLO': 0, 'DUO_CARRY': 1, 'DUO_SUPPORT': 2, 'DUO': 3, 'NONE': 4}
lane_dict = {'MID': 0, 'BOTTOM': 1, 'TOP': 2, 'JUNGLE': 3, 'NONE': 4}

mypymongo = MyPyMongo()
with open('../input/lol_basic.pickle', 'rb') as f:
    # M: number of champions, N: number of players
    champion_id2idx_dict, M, summoner_id2idx_dict, N = pickle.load(f)

# check if we need to continue unfinished job
# skip_match counts how many matches we have processed
if os.path.exists('../input/lol_lstm_match_hist.pickle'):
    with open('../input/lol_lstm_match_hist.pickle', 'rb') as f:
        match_hist, skip_match, feature_sum, feature_cnt = pickle.load(f)
else:
    match_hist = {}
    skip_match = 0
    # used for calculate feature mean
    feature_sum = numpy.zeros(len(useful_features), dtype=numpy.float64)
    feature_cnt = 0

start_time = time.time()

for cnt, match in enumerate(mypymongo.db.match_seed.find({}, skip=skip_match, no_cursor_timeout=True)):
    if cnt % 10 == 0:
        print("process {} matches, {} players, time {}"
              .format(skip_match + cnt, len(match_hist), time.time() - start_time))
        with open('../input/lol_lstm_match_hist.pickle', 'wb') as f:
            pickle.dump((match_hist, skip_match + cnt, feature_sum, feature_cnt), f)
        print('saved')

    # uncomment if you want to get a small dataset for test
    # if skip_match + cnt == 200:
    #     cnt -= 1
    #     break

    for participant in match['participants']:
        account_id = participant['accountId']
        if match_hist.get(account_id):
            continue
        # we want to find the match history for this player
        # 1522602430025 is the latest match in match_seed
        mh = mypymongo.find_match_history_in_match(account_id, 1522602430025)
        # two time features:
        # the first: time to last match,
        # the second : time to the current match (will be calculated in lstm training)
        time_vec = numpy.zeros((len(mh), 2))
        feature_vec = numpy.zeros((len(mh), len(useful_features)))
        role_vec = numpy.zeros((len(mh), len(role_dict)))
        lane_vec = numpy.zeros((len(mh), len(lane_dict)))
        champ_vec = numpy.zeros((len(mh), 1))
        for i, m in enumerate(mh):
            for j, k in enumerate(useful_features):
                # some features are missing in some mathes
                if m.get(k) is None:
                    m[k] = 0
                    print('match {} participant {} match {} key {} not found'
                          .format(match['id'], account_id, m['id'], k))
                if type(m[k]) is bool:
                    feature_vec[i, j] = int(m[k])
                else:
                    feature_vec[i, j] = m[k]
            # time feature 1: log(time) since last match
            if i == 0:
                # 1516100407349 is the earliest match in match db
                time_vec[i, 0] = numpy.log((m['timestamp'] - 1516100407349) / 1000 / 60 / 60)  # unit: log hour
            else:
                time_vec[i, 0] = numpy.log(
                    (m['timestamp'] - mh[i-1]['timestamp'] - mh[i-1]['duration'] * 1000) / 1000 / 60 / 60)
            # time feature 2 will be further calculated during lstm training
            # for now, just time stamp
            time_vec[i, 1] = m['timestamp']
            # one-hot encoding of role
            if not m['role']:
                role_vec[i, role_dict['NONE']] = 1
            else:
                role_vec[i, role_dict[m['role']]] = 1
            # one-hot encoding of lane
            if not m['lane']:
                lane_vec[i, lane_dict['NONE']] = 1
            else:
                lane_vec[i, lane_dict[m['lane']]] = 1
            # one-hot encoding of champion
            champ_vec[i, 0] = champion_id2idx_dict[m['champion_id']]

        # must put champ_vec in the first column, which will be used in lstm training
        match_hist[summoner_id2idx_dict[account_id]] = (champ_vec, role_vec, lane_vec, feature_vec, time_vec)
        # update for feature normalization
        feature_sum += numpy.sum(feature_vec, axis=0)
        feature_cnt += len(mh)

# post process.
# 1. calculate feature mean and normalize bt shifting feature values. We do not normalize for variance for now.
# 2. for each player, condense match history into a matrix (each row is the features of a match in his history)
feature_ave = feature_sum / feature_cnt
for k, v in match_hist.items():
    champ_vec, role_vec, lane_vec, feature_vec, time_vec = v
    assert feature_vec.shape[1] == feature_ave.shape[0]
    feature_vec = feature_vec / feature_ave
    match_hist[k] = numpy.hstack((champ_vec, role_vec, lane_vec, feature_vec, time_vec))

# save
with open('../input/lol_lstm_match_hist.pickle', 'wb') as f:
    pickle.dump((match_hist, skip_match + cnt + 1, feature_sum, feature_cnt), f)

print("total process time", time.time() - start_time)
print("number of summoner", len(match_hist))
