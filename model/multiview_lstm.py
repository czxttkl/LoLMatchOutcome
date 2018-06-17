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


def evaluate(y_test, y_pred):
    res = {}
    res['accuracy'] = float(sum(y_test == y_pred)) / len(y_test)
    res['precision'] = float(sum(y_test & y_pred) + 1) / (sum(y_pred) + 1)
    res['recall'] = float(sum(y_test & y_pred) + 1) / (sum(y_test) + 1)
    res['f_score'] = 2.0 * res['precision'] * res['recall'] / (res['precision'] + res['recall'])
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


def team_player_lstm_embed(train_match_data, test_match_data, match_hist_data, team,
                           mask_layer, gru_layer, drop_layer,
                           input_list, train_list, test_list):
    team_keys = []
    if team == 'red':
        team_keys = ['rp0', 'rp1', 'rp2', 'rp3', 'rp4']
    elif team == 'blue':
        team_keys = ['bp0', 'bp1', 'bp2', 'bp3', 'bp4']

    output_list = []
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
        mask_input = mask_layer(sub_input)
        sub_output = gru_layer(mask_input)
        drop_output = drop_layer(sub_output)
        output_list.append(drop_output)

    team_embed = merge(output_list, mode='concat', output_shape=(5*params['n_hidden'],))
    return team_embed


def team_champ_embed(champ_embed, champ_bias, train_match_data, test_match_data, team,
                     input_list, train_list, test_list):
    team_keys = []
    if team == 'red':
        team_keys = ['rc0', 'rc1', 'rc2', 'rc3', 'rc4']
    elif team == 'blue':
        team_keys = ['bc0', 'bc1', 'bc2', 'bc3', 'bc4']
    champ_input = Input(shape=(5,))
    input_list.append(champ_input)
    train_list.append(([[match_data[k] for k in team_keys] for match_data in train_match_data]))
    test_list.append(([[match_data[k] for k in team_keys] for match_data in test_match_data]))
    champ_embed = champ_embed(champ_input)
    champ_embed = Reshape((5 * params['n_hidden'],))(champ_embed)
    champ_bias = K.sum(champ_bias(champ_input))
    return champ_embed, champ_bias


def create_model(train_data, test_data, match_hist_data):
    input_list = []
    train_list = []
    test_list = []
    y_act = None
    objective = None

    seq_feat_num = train_data['rp0'][0].shape[1]
    mask_layer = Masking(mask_value=0, input_shape=(params['seq_max_len'], seq_feat_num))
    gru_layer = Bidirectional(GRU(output_dim=params['n_hidden'], return_sequences=False, consume_less='mem'))
    drop_layer = Dropout(params['dropout'])
    red_player_embed = team_player_lstm_embed(train_data, test_data, match_hist_data, 'red',
                                              mask_layer, gru_layer, drop_layer,
                                              input_list, train_list, test_list)
    blue_player_embed = team_player_lstm_embed(train_data, test_data, match_hist_data, 'blue',
                                               mask_layer, gru_layer, drop_layer,
                                               input_list, train_list, test_list)

    if params['idx'] == 2:
        champ_embed = Embedding(input_dim=params['champion_num'], output_dim=params['n_hidden'],
                                input_length=5, embeddings_initializer='uniform')
        champ_bias = Embedding(input_dim=params['champion_num'], output_dim=1,
                               input_length=5, embeddings_initializer='uniform')
        red_champ_embed, red_champ_bias = team_champ_embed(champ_embed, champ_bias, train_data, test_data, 'red',
                                                           input_list, train_list, test_list)
        blue_champ_embed, blue_champ_bias = team_champ_embed(champ_embed, champ_bias, train_data, test_data, 'blue',
                                                            input_list, train_list, test_list)
        red_player_champ_dot = dot([red_player_embed, red_champ_embed], axes=-1)
        blue_player_champ_dot = dot([blue_player_embed, blue_champ_embed], axes=-1)
        y = merge([red_player_champ_dot, blue_player_champ_dot, red_champ_bias, blue_champ_bias],
                  mode='concat', output_shape=(4,))
        # bias=True for red/blue team difference. the whole layer is set untrainable.
        y_act = Dense(1, activation='sigmoid', bias=True, kernel_initializer='ones', trainable=False)(y)
        objective = 'binary_crossentropy'
        metric = [acc]
    elif params['idx'] == 3:
        # set trainable to False as freeze layer.
        # see https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers
        Embedding(input_dim=16, output_dim=16, input_length=1, embeddings_initializer='identity', trainable=False)

    model = Model(input=input_list, output=y_act)
    model.compile(loss=objective, optimizer=RMSprop(lr=params['lr']), metrics=metric)
    return model, train_list, test_list


def run_model(train_data, test_data, y_train, y_test, match_hist_data):
    model, X_train, X_test = create_model(train_data, test_data, match_hist_data)
    hist = model.fit(x=X_train, y=y_train, batch_size=params['batch_size'], verbose=2,
                     nb_epoch=params['n_epochs'], validation_data=(X_test, y_test))
    y_score = model.predict(X_test, batch_size=params['batch_size'], verbose=0)
    y_pred = (np.ravel(y_score) > 0.5).astype('int32')
    return y_pred, hist.history


def train():
    reader = CVFoldDenseReader(data_path=params['match_path'], folds=params['fold'])
    with open(params['match_hist_path'], 'rw') as f:
        match_hist_data = pickle.load(f)

    for fold in range(params['fold']):
        # original data format: match_num x each dict contains id of players and champions
        train_data, y_train, test_data, y_test = reader.read_train_test_fold(fold)
        y_pred, hist = run_model(train_data, test_data, y_train, y_test, match_hist_data)
        res = evaluate(y_test, y_pred)


if __name__ == '__main__':
    with open('../input/lol_basic.pickle', 'rb') as f:
        # M: number of champions, N: number of players
        champion_id2idx_dict, M, summoner_id2idx_dict, N = pickle.load(f)

    params = {
              'champion_num': M,
              'match_hist_path': 'input/lstm_match_hist.lol',
              'match_path': 'input/lstm_match.lol',
              'fold': 10,
              'seq_min_len': 10,
              'seq_max_len': 1500,
              'batch_size': 256,
              'lr': 0.001,
              'dropout': 0.0,
              'n_epochs': 500,
              'n_hidden': 8,
              # 'n_latent': 8,
              'n_classes': 1,
              # 'bias': 1,
              # 'is_clf': 1,
              'idx': 2,  # 1: player average, 2: player-champion dot, 3: dmvm
              }
    train()
