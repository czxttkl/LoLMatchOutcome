'''
neural network baseline
'''
import sys
sys.path.insert(0, '..')

from baseline import Baseline
from sklearn.linear_model import LogisticRegression
from data_mangle.report_writer import ReportWriter
from utils.parser import parse_ml_parameters
from data_mangle.cv_fold_dense_reader import CVFoldDenseReader


class BaselineNN(Baseline):

    def print_model(self, model):
        """ return description of a model as a string """
        return "LR_ds{}_fs{}".format(model.train_data_size, model.feature_size)


if __name__ == "__main__":
    kwargs = parse_ml_parameters()

    dataset = 'lol_player_champion_nn' if not kwargs else kwargs.dataset

    data_path = '../input/{}.pickle'.format(dataset)
    print('use parameter: dataset {}, '.format(dataset))

    baseline = \
        BaselineNN(
            models=[LogisticRegression(fit_intercept=True),
                    # add more grid search models here ...
                    ],
            reader=CVFoldDenseReader(data_path=data_path, folds=10),
            writer=ReportWriter('result.csv'))
    baseline.cross_valid()

    # baseline.save_model()
