'''
neural network baseline
'''
import sys
sys.path.insert(0, '..')

from baseline import Baseline
from sklearn.neural_network import MLPClassifier
from data_mangle.report_writer import ReportWriter
from utils.parser import parse_ml_parameters, parse_reader


class BaselineNN(Baseline):

    def print_model(self, model):
        """ return description of a model as a string """
        return "NN_hiddenunit{}".format(model.hidden_layer_sizes[0])


if __name__ == "__main__":
    kwargs = parse_ml_parameters()

    dataset = 'dota' if not kwargs else kwargs.dataset
    density = 'dense' if not kwargs else kwargs.density

    feature_config = 'one_way_one_team' if not kwargs else kwargs.nn_featconfig
    nn_hidden = 100 if not kwargs else kwargs.nn_hidden

    print('use parameter: dataset {}, feature_config: {}, density: {}, nn_hidden: {}'
          .format(dataset, feature_config, density, nn_hidden))
    reader = parse_reader(dataset, feature_config, density)

    baseline = \
        BaselineNN(
            models=[MLPClassifier(hidden_layer_sizes=(nn_hidden,),),
                    # add more grid search models here ...
                    ],
            # reader=CVFoldLoLSparseReader(data_path=constants.lol_pickle, folds=10,
            #                              feature_config='champion_summoner_one_team'),
            # reader=CVFoldSparseReader(data_path=constants.dota_pickle, folds=10,
            #                           feature_config='one_way_one_team'),
            reader=reader,
            writer=ReportWriter('result.csv'))
    baseline.cross_valid()

    # baseline.save_model()
