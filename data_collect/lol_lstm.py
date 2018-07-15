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
        if cnt % 10 == 0:
            print("process {} match".format(cnt))
        if cnt == 200:
            break

        x = {}
        r_cnt, b_cnt = 0, 0
        invalid_participant = False
        for participant in match['participants']:
            # 1522602430025 is the latest match in match_seed
            match_hist = mypymongo.find_match_history_in_match(participant['accountId'], 1522602430025)
            if len(match_hist) == 0:
                invalid_participant = True
                print('find invalid participants with no match history', participant['accountId'], match['id'])
                break

            if participant['side'] == 'red':
                x['rp' + str(r_cnt)] = summoner_id2idx_dict[participant['accountId']]
                x['rc' + str(r_cnt)] = champion_id2idx_dict[participant['championId']]
                r_cnt += 1
            else:
                x['bp' + str(b_cnt)] = summoner_id2idx_dict[participant['accountId']]
                x['bc' + str(b_cnt)] = champion_id2idx_dict[participant['championId']]
                b_cnt += 1

        if invalid_participant:
            print("invalid participant due to zero match history")
            continue

        # timestamp
        x['t'] = match['gameCreation']
        X.append(x)

        for team in match['teams']:
            if team['side'] == 'red':
                if team['isWinner']:
                    Y.append(1)
                else:
                    Y.append(0)

    return X, Y


if __name__ == "__main__":
    start_time = time.time()

    mypymongo = MyPyMongo()

    X, Y = get_data()

    with open("../input/lol_lstm_match.pickle", "wb") as f:
        pickle.dump((X, Y), f)

    print("total process time", time.time() - start_time)


