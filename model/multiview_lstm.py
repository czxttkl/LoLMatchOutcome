'''
Adapted from https://github.com/caobokai/DeepMood/blob/master/DeepMood.py
A demo of DeepMood on synthetic data.
[bib] DeepMood: Modeling Mobile Phone Typing Dynamics for Mood Detection, B. Cao et al., 2017.
'''

from __future__ import print_function
import random
from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.normalization import *
from data_mangle.cv_fold_dense_reader import CVFoldDenseReader
from keras.preprocessing import sequence
import pickle


def evaluate(y_test, y_pred, params):
    res = {}
    if params['is_clf']:
        res['accuracy'] = float(sum(y_test == y_pred)) / len(y_test)
        res['precision'] = float(sum(y_test & y_pred) + 1) / (sum(y_pred) + 1)
        res['recall'] = float(sum(y_test & y_pred) + 1) / (sum(y_test) + 1)
        res['f_score'] = 2.0 * res['precision'] * res['recall'] / (res['precision'] + res['recall'])
    else:
        res['rmse'] = np.sqrt(np.mean(np.square(y_test - y_pred)))
        res['mae'] = np.mean(np.abs(y_test - y_pred))
        res['explained_variance_score'] = 1 - np.square(np.std(y_test - y_pred)) / np.square(np.std(y_test))
        res['r2_score'] = 1 - np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test)))
    print(' '.join(["%s: %.4f" % (i, res[i]) for i in res]))
    return res


def mvm_decision_function(arg):
    n_modes = len(arg)
    latentx = arg
    y = K.concatenate([K.sum(
        K.prod(K.stack([latentx[j][:, i * params['n_latent']: (i + 1) * params['n_latent']] for j in range(n_modes)]),
               axis=0),
        axis=-1, keepdims=True) for i in range(params['n_classes'])])
    return y


def fm_decision_function(arg):
    latentx, bias = arg[0], arg[1]
    pairwise = K.sum(K.square(latentx), axis=-1, keepdims=True)
    y = K.sum(K.tf.stack([pairwise, bias]), axis=0)
    return y


def acc(y_true, y_pred):
    return K.mean(K.equal(y_true > 0.5, y_pred > 0.5))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def team_lstm_embed(train_match_data, test_match_data, match_hist_data, team,
                    input_list, train_list, test_list):
    if team == 'red':
        team_keys = ['rp0', 'rp1', 'rp2', 'rp3', 'rp4']
    elif team == 'blue':
        team_keys = ['bp0', 'bp1', 'bp2', 'bp3', 'bp4']

    output_list = []
    K.zeros()
    seq_feat_num = train_match_data['rp0'][0].shape[1]

    for p in team_keys:
        sub_input = Input(shape=(params['seq_max_len'], seq_feat_num))
        input_list.append(sub_input)
        train_data_p = sequence.pad_sequences([match_hist_data[match_data[p]] for match_data in train_match_data],
                                              maxlen=params['seq_max_len'])
        test_data_p = sequence.pad_sequences([match_hist_data[match_data[p]] for match_data in test_match_data],
                                             maxlen=params['seq_max_len'])
        train_list.append(train_data_p)
        test_list.append(test_data_p)
        mask_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], seq_feat_num))(sub_input)
        sub_output = Bidirectional(
            GRU(output_dim=params['n_hidden'], return_sequences=False, consume_less='mem'))(mask_input)
        drop_output = Dropout(params['dropout'])(sub_output)
        output_list.append(drop_output)
    return output_list


def create_model(train_data, test_data, match_hist_data, params):
    input_list = []
    train_list = []
    test_list = []

    if params['idx'] == 2:
        red_team_output_list = team_lstm_embed(train_data, test_data, match_hist_data, 'red',
                                               input_list, train_list, test_list)
        blue_team_output_list = team_lstm_embed(train_data, test_data, match_hist_data, 'blue',
                                                input_list, train_list, test_list)

        x = merge(output_list, mode='concat') if len(output_list) > 1 else output_list[0]
        latentx = Dense(params['n_latent'], bias=False)(x)
        bias = Dense(1, bias=True if params['bias'] else False)(x)
        y = merge([latentx, bias], mode=fm_decision_function, output_shape=(1,))
        y_act = Activation('sigmoid')(y) if params['is_clf'] else Activation('linear')(y)
        objective = 'binary_crossentropy' if params['is_clf'] else 'mean_squared_error'
        metric = [acc] if params['is_clf'] else [rmse]

    model = Model(input=input_list, output=y_act)
    model.compile(loss=objective, optimizer=RMSprop(lr=params['lr']), metrics=metric)
    return model, train_list, test_list


def run_model(train_data, test_data, y_train, y_test, match_hist_data, params):
    model, X_train, X_test = create_model(train_data, test_data, match_hist_data, params)
    hist = model.fit(x=X_train, y=y_train, batch_size=params['batch_size'], verbose=2,
                     nb_epoch=params['n_epochs'], validation_data=(X_test, y_test))
    y_score = model.predict(X_test, batch_size=params['batch_size'], verbose=0)
    y_pred = (np.ravel(y_score) > 0.5).astype('int32') if params['is_clf'] else np.ravel(y_score)
    return y_pred, hist.history


params = {
          'match_hist_path': 'input/lstm_match_hist.lol',
          'match_path': 'input/lstm_match.lol',
          'fold': 10,
          'seq_min_len': 10,
          'seq_max_len': 2000,
          'batch_size': 256,
          'lr': 0.001,
          'dropout': 0.0,
          'n_epochs': 500,
          'n_hidden': 8,
          # 'n_latent': 8,
          'n_classes': 1,
          'bias': 1,
          'is_clf': 1,
          'idx': 2,  # 1: dnn, 2: dfm, 3: dmvm
          }

reader = CVFoldDenseReader(data_path=params['match_path'], folds=params['fold'])
with open(params['match_hist_path'], 'rw') as f:
    match_hist_data = pickle.load(f)

for fold in range(params['fold']):
    # original data format: match_num x each dict contains id of players and champions
    train_data, y_train, test_data,  y_test = reader.read_train_test_fold(fold)
    y_pred, hist = run_model(train_data, test_data, y_train, y_test, match_hist_data, params)
    res = evaluate(y_test, y_pred, params)
