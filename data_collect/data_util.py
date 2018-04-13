import numpy


def ave_stats(stats, stat_names: list):
    feature = numpy.zeros(len(stat_names))

    if not stats:
        return feature

    for stat in stats:
        feature += numpy.array([stat[stat_name] for stat_name in stat_names])

    feature /= len(stats)
    return feature


def sum_stats(stats, stat_names: list):
    feature = numpy.zeros(len(stat_names))

    if not stats:
        return feature

    for stat in stats:
        feature += numpy.array([stat[stat_name] for stat_name in stat_names])

    return feature


def if_red_team_win(match):
    if match['teams'][0]['side'] == 'blue':
        if match['teams'][0]['isWinner']:
            return False
        else:
            return True
    else:
        if match['teams'][0]['isWinner']:
            return True
        else:
            return False
