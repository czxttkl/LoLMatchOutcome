'''
neural network baseline
'''
import sys
sys.path.insert(0, '..')

from baseline import Baseline
from sklearn.neural_network import MLPClassifier
from data_mangle.report_writer import ReportWriter
from utils.parser import parse_ml_parameters
from data_mangle.cv_fold_dense_reader import CVFoldDenseReader


class BaselineNN(Baseline):

    def print_model(self, model):
        """ return description of a model as a string """
        return "NN_hiddenunit{}_ds{}_fs{}".format(model.hidden_layer_sizes[0],
                                                  model.train_data_size, model.feature_size)


if __name__ == "__main__":
    kwargs = parse_ml_parameters()

    dataset = 'lol_player_champion_nn' if not kwargs else kwargs.dataset
    nn_hidden = 3 if not kwargs else kwargs.nn_hidden

    data_path = '../input/{}.pickle'.format(dataset)
    print('use parameter: dataset {}, nn_hidden: {}'.format(dataset, nn_hidden))

    baseline = \
        BaselineNN(
            models=[MLPClassifier(hidden_layer_sizes=(nn_hidden,)),
                    # add more grid search models here ...
                    ],
            reader=CVFoldDenseReader(data_path=data_path, folds=1, seed=715),
            writer=ReportWriter('result.csv'))
    baseline.cross_valid()

    # baseline.save_model()
