from pymongo import MongoClient
import pymongo

class MyPyMongo:

    def __init__(self):
        client = MongoClient('localhost', 27017)
        self.db = client['lol']

    def find_match_history_in_match(self, account_id, before_time, champion_id=None):
        player_stats = []
        query_dict = {'accountId': account_id,
                      'timestamp': {'$lt': before_time},
                      'season': 11,
                      'queue': 420,
                      }
        if champion_id is not None:
             query_dict['champion'] = champion_id

        for match_history in self.db.player_seed_match_history.find(query_dict) \
                .sort([("timestamp", pymongo.ASCENDING)]):
            match_id = match_history['gameId']
            match = self.db.match.find_one({'id': match_id})
            # only legitimately long match
            if match['duration'] < 300:
                continue
            for participant in match['participants']:
                if participant['accountId'] == account_id:
                    player_stat = participant['stats']
                    player_stat['role'] = match_history['role']
                    player_stat['lane'] = match_history['lane']
                    player_stat['presence'] = 1
                    player_stat['timestamp'] = match['gameCreation']
                    player_stat['duration'] = match['gameDuration']
                    player_stat['id'] = match['id']
                    player_stats.append(player_stat)
                    break
        return player_stats

    def exist_match_id_in_match_seed(self, match_id: int):
        match_id = int(match_id)
        if self.db.match_seed.find({'id': match_id}).count() > 0:
            return True
        else:
            return False

    def exist_match_id_in_match(self, match_id: int):
        match_id = int(match_id)
        if self.db.match.find({'id': match_id}).count() > 0:
            return True
        else:
            return False

    def exist_account_id_in_player_seed(self, account_id: int):
        account_id = int(account_id)
        if self.db.player_seed.find({'accountId': account_id}).count() > 0:
            return True
        else:
            return False

    def find_player_in_player_seed(self, account_id: int):
        return self.db.player_seed.find_one({'accountId': account_id})
