"""
Use this file to generate data for lstm
"""
import pickle
import time
from data_collect.mypymongo import MyPyMongo


def update_dict(d: dict, key):
    if d.get(key) is None:
        key_id = len(d)
        d[key] = key_id
    return d


def get_data():
    with open('../input/lol_basic.pickle', 'rb') as f:
        # M: number of champions, N: number of players
        champion_id2idx_dict, M, summoner_id2idx_dict, N = pickle.load(f)

    X, Y = [], []
    for cnt, match in enumerate(mypymongo.db.match_seed.find({}, no_cursor_timeout=True)):
        if cnt % 100 == 0:
            print("process {} match".format(cnt))
        if cnt == 1000:
            break

        for team in match['teams']:
            if team['side'] == 'red':
                if team['isWinner']:
                    Y.append(1)
                else:
                    Y.append(0)
        x = {}
        r_cnt, b_cnt = 0, 0
        for participant in match['participants']:
            if participant['side'] == 'red':
                x['rp' + str(r_cnt)] = summoner_id2idx_dict[participant['accountId']]
                x['rc' + str(r_cnt)] = champion_id2idx_dict[participant['championId']]
                r_cnt += 1
            else:
                x['bp' + str(b_cnt)] = summoner_id2idx_dict[participant['accountId']]
                x['bc' + str(b_cnt)] = champion_id2idx_dict[participant['championId']]
                b_cnt += 1
        # timestamp
        x['t'] = match['gameCreation']
        X.append(x)

    return X, Y


if __name__ == "__main__":
    start_time = time.time()

    mypymongo = MyPyMongo()

    X, Y = get_data()

    with open("../input/lol_lstm.pickle", "wb") as f:
        pickle.dump((X, Y), f)

    print("total process time", time.time() - start_time)


