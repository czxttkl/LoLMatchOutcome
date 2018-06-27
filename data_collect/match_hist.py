"""
Use this file to generate player history feature
"""
import pickle
import time
from data_collect.mypymongo import MyPyMongo
import numpy

useless_features = {
    'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'participantId',
    'perkPrimaryStyle', 'perkSubStyle', 'presence',
    'playerScore0', 'playerScore1', 'playerScore2', 'playerScore3', 'playerScore4',
    'playerScore5', 'playerScore6', 'playerScore7', 'playerScore8', 'playerScore9',
}

# exclude role and lane
useful_features = [
 'assists',
 'champLevel',
 'combatPlayerScore',
 'damageDealtToObjectives',
 'damageDealtToTurrets',
 'damageSelfMitigated',
 'deaths',
 'doubleKills',
 'duration',
 'firstBloodAssist',
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
 'sightWardsBoughtInGame',
 'timeCCingOthers',
 'totalDamageDealt',
 'totalDamageDealtToChampions',
 'totalDamageTaken',
 'totalHeal',
 'totalMinionsKilled',
 'totalPlayerScore',
 'totalScoreRank',
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

start_time = time.time()
mypymongo = MyPyMongo()
match_hist = {}
with open('../input/lol_basic.pickle', 'rb') as f:
    # M: number of champions, N: number of players
    champion_id2idx_dict, M, summoner_id2idx_dict, N = pickle.load(f)

for cnt, match in enumerate(mypymongo.db.match_seed.find({}, no_cursor_timeout=True)):
    if cnt % 100 == 0:
        print("process {} match".format(cnt))
    if cnt == 1000:
        break

    for participant in match['participants']:
        account_id = participant['accountId']
        if match_hist.get(account_id):
            continue
        # 1522602430025 is the latest match in match_seed
        mh = mypymongo.find_match_history_in_match(account_id, 1522602430025)
        # two more features:
        # the second to last: time to last match,
        # last: time to the current match (will be calculated in lstm training)
        feature_vec = numpy.zeros((len(mh), len(useful_features) + 2))
        role_vec = numpy.zeros((len(mh), len(role_dict)))
        lane_vec = numpy.zeros((len(mh), len(lane_dict)))
        champ_vec = numpy.zeros((len(mh), 1))
        for i, m in enumerate(mh):
            for j, k in enumerate(useful_features):
                if m.get(k) is None:
                    m[k] = 0
                    print('match {} participant {} match {} key {} not found'
                          .format(match['id'], account_id, m['id'], k))
                if type(m[k]) is bool:
                    feature_vec[i, j] = int(m[k])
                else:
                    feature_vec[i, j] = m[k]
            if i == 0:
                # 1516100407349 is the earliest match in match db
                feature_vec[i, -2] = numpy.log((m['timestamp'] - 1516100407349) / 1000 / 60 / 60)  # unit: log hour
            else:
                feature_vec[i, -2] = numpy.log(
                    (m['timestamp'] - mh[i-1]['timestamp'] - mh[i-1]['duration'] * 1000) / 1000 / 60 / 60)
            # the last colume will be further calculated during lstm training
            # for now, just time stamp
            feature_vec[i, -1] = m['timestamp']
            if not m['role']:
                role_vec[i, role_dict['NONE']] = 1
            else:
                role_vec[i, role_dict[m['role']]] = 1
            if not m['lane']:
                lane_vec[i, lane_dict['NONE']] = 1
            else:
                lane_vec[i, lane_dict[m['lane']]] = 1
            champ_vec[i, 0] = m['champion_id']

        match_hist[summoner_id2idx_dict[account_id]] = numpy.hstack((champ_vec, role_vec, lane_vec, feature_vec))

with open('../input/match_hist.pickle', 'wb') as f:
    pickle.dump(match_hist, f)

print("total process time", time.time() - start_time)
print("number of summoner", len(match_hist))
