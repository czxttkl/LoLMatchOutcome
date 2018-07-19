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

    with open('../input/lol_match_id_record.pickle', 'rb') as f:
        match_id_record = pickle.load(f)

    match_id2data = {}
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

        assert r_cnt == b_cnt == 5

        # timestamp
        x['t'] = match['gameCreation']

        for team in match['teams']:
            if team['side'] == 'red':
                if team['isWinner']:
                    y = 1
                else:
                    y = 0

        match_id2data[match['id']] = [x, y]

    print()
    match_data = {}
    for fold in match_id_record:
        match_data[fold] = {}
        for dataset in match_id_record[fold]:
            match_data[fold][dataset] = [[], []]
            print('in match_id_record, fold: {}, dataset: {}, # of matches: {}'
                  .format(fold, dataset, len(match_id_record[fold][dataset])))
            for match_id in match_id_record[fold][dataset]:
                if match_id in match_id2data:
                    match_data[fold][dataset][0].append(match_id2data[match_id][0])
                    match_data[fold][dataset][1].append(match_id2data[match_id][1])
            print('match_data, fold: {}, dataset: {}, # of matches: {}'
                  .format(fold, dataset, len(match_data[fold][dataset][1])))

    return match_data


if __name__ == "__main__":
    start_time = time.time()

    mypymongo = MyPyMongo()

    match_data = get_data()

    with open("../input/lol_lstm_match.pickle", "wb") as f:
        pickle.dump(match_data, f)

    print("total process time", time.time() - start_time)


