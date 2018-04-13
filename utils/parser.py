import optparse
import sys


def parse_ml_parameters():
    if len(sys.argv) < 2:
        print('no argument set. use default.')
        return None

    parser = optparse.OptionParser(usage="usage: %prog [options]")
    # NN
    parser.add_option("--nn_hidden", dest="nn_hidden", type="int", default=0)
    # general
    parser.add_option("--dataset", dest='dataset', type='string', default='')
    (kwargs, args) = parser.parse_args()
    return kwargs

