"""
generate basic information of lol 
"""
from mypymongo import MyPyMongo
import pickle

def update_dict(d: dict, key):
    if d.get(key) is None:
        key_id = len(d)
        d[key] = key_id
    return d

mypymongo = MyPyMongo()
champion_id2idx_dict, summoner_id2idx_dict = {}, {}

cnt = 0
for match in mypymongo.db.match_seed.find({}, no_cursor_timeout=True):
    cnt += 1
    if cnt % 100 == 0:
        print("process {} match".format(cnt))

    for participant in match['participants']:
        update_dict(summoner_id2idx_dict, participant['accountId'])
        update_dict(champion_id2idx_dict, participant['championId'])

N = len(summoner_id2idx_dict)
M = len(champion_id2idx_dict)

print('M: {}, N: {}\nchampion_id2idx_dict:{}'
      .format(M, N, champion_id2idx_dict))

with open('../input/lol_basic.pickle', 'wb') as f:
    # M: number of champions, N: number of players
    pickle.dump((champion_id2idx_dict, M, summoner_id2idx_dict, N), f)