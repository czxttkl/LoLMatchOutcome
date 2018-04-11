"""
This file quickly verifies that whether the sum of win rate on selected champion is a good indicator
See Thomas Huang's technical report: https://thomasythuang.github.io/League-Predictor/abstract.pdf
"""

from data_collect.mypymongo import MyPyMongo
import pprint
from data_util import ave_stats, if_red_team_win


if __name__ == "__main__":
    mypymongo = MyPyMongo()

    total_match = 0
    correct_match = 0

    for match in mypymongo.db.match_seed.find({'all_participants_matches_crawled': True}):
        match_create_time = match['gameCreation']
        match_id = match['id']
        red_team_win = if_red_team_win(match)
        red_team_win_rate = 0
        blue_team_win_rate = 0

        for participant in match['participants']:
            account_id = participant['accountId']
            champion_id = participant['championId']
            side = participant['side']
            player_stats = mypymongo.find_match_history_in_match(account_id, match_create_time, champion_id=champion_id)
            feature = ave_stats(player_stats, ['win'])[0]
            if feature == 0:
                feature = 0.5

            if side == 'blue':
                blue_team_win_rate += feature
            else:
                red_team_win_rate += feature

        print('match {}. red WR: {:.5f}. blue WR: {:.5f}. red team win: {}'
              .format(total_match, red_team_win_rate, blue_team_win_rate, red_team_win))

        total_match += 1
        if (red_team_win_rate > blue_team_win_rate and red_team_win)\
                or (red_team_win_rate < blue_team_win_rate and not red_team_win):
            correct_match += 1

    print('total_match {}, correct_match {}, accuracy {}'
          .format(total_match, correct_match, correct_match / total_match))
