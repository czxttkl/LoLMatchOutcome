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


def parse_lstm_parameters(default_params):
    if len(sys.argv) < 2:
        print('no argument set. use default.')
        return default_params

    parser = optparse.OptionParser(usage="usage: %prog [options]")

    parser.add_option("--champion_num", dest="champion_num", type="int",
                      default=default_params['champion_num'])
    parser.add_option("--match_hist_path", dest="match_hist_path", type="string",
                      default=default_params['match_hist_path'])
    parser.add_option("--match_path", dest="match_path", type="string",
                      default=default_params['match_path'])
    parser.add_option("--fold", dest="fold", type="int",
                      default=default_params['fold'])
    parser.add_option("--seq_max_len", dest="seq_max_len", type="int",
                      default=default_params['seq_max_len'])
    parser.add_option("--batch_size", dest="batch_size", type="int",
                      default=default_params['batch_size'])
    parser.add_option("--lr", dest="lr", type="double",
                      default=default_params['lr'])
    parser.add_option("--dropout", dest="dropout", type="double",
                      default=default_params['dropout'])
    parser.add_option("--batch_size", dest="batch_size", type="int",
                      default=default_params['batch_size'])
    parser.add_option("--n_epochs", dest="n_epochs", type="int",
                      default=default_params['n_epochs'])
    parser.add_option("--n_hidden", dest="n_hidden", type="int",
                      default=default_params['n_hidden'])
    parser.add_option("--n_hidden1", dest="n_hidden1", type="int",
                      default=default_params['n_hidden1'])

    # model index
    # 1: player lstm average,
    # 2: player lstm - champion embedding dot product,
    # 3: player contextualized by champion one-hot
    # 4: 3 + synergy + opposition
    # 5: fully-connected nn, each champion has a lstm block, then two teams sum
    # 6: fully-connected nn, each champion has a lstm block,
    # then form a team embedding, then two team embedding sum
    parser.add_option("--idx", dest='idx', type='int',
                      default=default_params['idx'])

    (kwargs, args) = parser.parse_args()
    return kwargs

