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
from keras.preprocessing import sequence
import pickle
import numpy


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


def transform_match_hist(mh, mt):
    # convert champion into one hot encoding
    champ_vecs = numpy.zeros((len(mh), params['champion_num']))
    # the first column of mh is champion id
    champ_vecs[:, mh[:, 0].astype(int)] = 1
    transformed = numpy.hstack((champ_vecs, mh[:, 1:]))
    # only matches before match time (mt)
    transformed = transformed[transformed[:, -1] < mt, :]
    # truncate rows to params['seq_max_len']
    transformed = transformed[:params['seq_max_len'], :]
    # time transform to log
    transformed[:, -1] = numpy.log((mt - transformed[:, -1]) / 1000 / 60 / 60)
    return transformed


def team_player_lstm_embed(train_match_data, test_match_data, match_hist_data, team,
                           mask_layer, gru_layer, drop_layer,
                           input_list, train_list, test_list):
    team_keys = []
    if team == 'red':
        team_keys = ['rp0', 'rp1', 'rp2', 'rp3', 'rp4']
    elif team == 'blue':
        team_keys = ['bp0', 'bp1', 'bp2', 'bp3', 'bp4']

    output_list = []

    for p in team_keys:
        p_hist_input = Input(shape=(params['seq_max_len'], params['seq_feat_num']))
        input_list.append(p_hist_input)
        train_data_p = sequence.pad_sequences([transform_match_hist(
                                                match_hist_data[match_data[p]], match_data['t']
                                               ) for match_data in train_match_data],
                                              maxlen=params['seq_max_len'])
        test_data_p = sequence.pad_sequences([transform_match_hist(
                                               match_hist_data[match_data[p]], match_data['t']
                                               ) for match_data in test_match_data],
                                             maxlen=params['seq_max_len'])
        train_list.append(train_data_p)
        test_list.append(test_data_p)
        mask_output = mask_layer(p_hist_input)
        gru_output = gru_layer(mask_output)
        drop_output = drop_layer(gru_output)
        output_list.append(drop_output)

    return output_list


def team_champ_embed(champ_embed, champ_bias, train_match_data, test_match_data, team,
                     input_list, train_list, test_list):
    team_keys = []
    if team == 'red':
        team_keys = ['rc0', 'rc1', 'rc2', 'rc3', 'rc4']
    elif team == 'blue':
        team_keys = ['bc0', 'bc1', 'bc2', 'bc3', 'bc4']
    champ_input = Input(shape=(5,))
    input_list.append(champ_input)
    train_list.append(numpy.array(
        [[match_data[k] for k in team_keys] for match_data in train_match_data]))
    test_list.append(numpy.array(
        [[match_data[k] for k in team_keys] for match_data in test_match_data]))
    champ_embed_output = champ_embed(champ_input)
    champ_bias_output = champ_bias(champ_input)
    return champ_embed_output, champ_bias_output


def team_player_champ_lstm_embed(train_match_data, test_match_data, match_hist_data, team,
                                 mask_layer, gru_layer, drop_layer, champ_one_hot, player_champ_matrix,
                                 input_list, train_list, test_list):
    team_keys = []
    if team == 'red':
        team_keys = [('rp0', 'rc0'), ('rp1', 'rc1'), ('rp2', 'rc2'), ('rp3', 'rc3'), ('rp4', 'rc4')]
    elif team == 'blue':
        team_keys = [('bp0', 'bc0'), ('bp1', 'bc1'), ('bp2', 'bc2'), ('bp3', 'bc3'), ('bp4', 'bc4')]

    output_list = []

    for p, c in team_keys:
        p_hist_input = Input(shape=(params['seq_max_len'], params['seq_feat_num']))
        input_list.append(p_hist_input)
        train_data_p = sequence.pad_sequences([transform_match_hist(
                                               match_hist_data[match_data[p]], match_data['t']
                                               ) for match_data in train_match_data],
                                              maxlen=params['seq_max_len'])
        test_data_p = sequence.pad_sequences([transform_match_hist(
                                               match_hist_data[match_data[p]], match_data['t']
                                               ) for match_data in test_match_data],
                                              maxlen=params['seq_max_len'])
        train_list.append(train_data_p)
        test_list.append(test_data_p)
        mask_output = mask_layer(p_hist_input)
        gru_output = gru_layer(mask_output)
        drop_output = drop_layer(gru_output)

        c_input = Input(shape=(1,))
        input_list.append(c_input)
        train_list.append([match_data[c] for match_data in train_match_data])
        test_list.append([match_data[c] for match_data in test_match_data])
        champ_one_hot_output = champ_one_hot(c_input)
        # output shape: 2 * params['n_hidden'] + params['champion_num']
        dropout_champ_one_hot = Concatenate()([drop_output, champ_one_hot_output])
        player_champ_output = player_champ_matrix(dropout_champ_one_hot)
        output_list.append(player_champ_output)

    return output_list


def team_player_champ_lstm_nn(train_match_data, test_match_data, match_hist_data, team,
                              mask_layer, gru_layer, drop_layer, champ_one_hot,
                              input_list, train_list, test_list):
    team_keys = []
    if team == 'red':
        team_keys = [('rp0', 'rc0'), ('rp1', 'rc1'), ('rp2', 'rc2'), ('rp3', 'rc3'), ('rp4', 'rc4')]
    elif team == 'blue':
        team_keys = [('bp0', 'bc0'), ('bp1', 'bc1'), ('bp2', 'bc2'), ('bp3', 'bc3'), ('bp4', 'bc4')]

    p_output_list = []
    c_output_list = []

    for p, c in team_keys:
        p_hist_input = Input(shape=(params['seq_max_len'], params['seq_feat_num']))
        input_list.append(p_hist_input)
        train_data_p = sequence.pad_sequences([transform_match_hist(
                                                match_hist_data[match_data[p]]
                                                ) for match_data in train_match_data],
                                              maxlen=params['seq_max_len'])
        test_data_p = sequence.pad_sequences([transform_match_hist(
                                                match_hist_data[match_data[p]]
                                                ) for match_data in test_match_data],
                                             maxlen=params['seq_max_len'])
        train_list.append(train_data_p)
        test_list.append(test_data_p)
        mask_output = mask_layer(p_hist_input)
        gru_output = gru_layer(mask_output)
        drop_output = drop_layer(gru_output)
        p_output_list.append(drop_output)

        c_input = Input(shape=(1,))
        input_list.append(c_input)
        train_list.append([match_data[c] for match_data in train_match_data])
        test_list.append([match_data[c] for match_data in test_match_data])
        champ_one_hot_output = champ_one_hot(c_input)
        c_output_list.append(champ_one_hot_output)

    # [[0, 1],                        [[1,2,3],
    #  [1, 0],       [[4,5,6],         [4,5,6],
    #  [0, 0],   *    [1,2,3]]    =    [0,0,0],
    #  [0, 0],                         [0,0,0],
    #  [0, 0]]                         [0,0,0]]
    player_embed_mat = K.tf.stack(p_output_list)
    champ_onehot_mat = K.tf.transpose(K.tf.stack(c_output_list))
    out = K.tf.matmul(champ_onehot_mat, player_embed_mat)
    out_flat = Reshape((params['champion_num'] * 2 * params['n_hidden'],))(out)
    # concatenate champion one hot encoding so that we can capture pure champion interaction
    # output shape: params['champion_num']
    champ_onehot_vec = Add()(c_output_list)
    # output shape: (2 * params['n_hidden'] + 1) * params['champion_num']
    out_final = Concatenate()([out_flat, champ_onehot_vec])
    return out_final


def synergy(team_embeds, syng_mat):
    # output_shape: params['n_hidden']
    team_embed_vec = Add()(team_embeds)
    a = dot([syng_mat(team_embed_vec), team_embed_vec], axes=-1)
    for team_embed in team_embeds:
        b = dot([syng_mat(team_embed), team_embed], axes=-1)
        a = subtract([a, b])
    return a


def opposition(red_team_embeds, blue_team_embeds, oppo_mat):
    # output_shape: params['n_hidden']
    red_team_embed_vec = Add()(red_team_embeds)
    blue_team_embed_vec = Add()(blue_team_embeds)
    a = dot([oppo_mat(red_team_embed_vec), blue_team_embed_vec], axes=-1)
    b = dot([oppo_mat(blue_team_embed_vec), red_team_embed_vec], axes=-1)
    c = subtract([a, b])
    return c


def create_model(train_data, test_data, match_hist_data):
    input_list = []
    train_list = []
    test_list = []
    y_act = None

    mask_layer = Masking(mask_value=0, input_shape=(params['seq_max_len'], params['seq_feat_num']))
    # bidirectional output dimension will double, i.e., output shape: 2 * params['n_hidden']
    gru_layer = Bidirectional(GRU(return_sequences=False, units=params['n_hidden'], implementation=1))
    drop_layer = Dropout(params['dropout'])

    if params['idx'] == 1:
        # shape = (, 2 * params['n_hidden']) if bidirectional
        red_player_embeds = team_player_lstm_embed(train_data, test_data, match_hist_data, 'red',
                                                   mask_layer, gru_layer, drop_layer,
                                                   input_list, train_list, test_list)
        blue_player_embeds = team_player_lstm_embed(train_data, test_data, match_hist_data, 'blue',
                                                    mask_layer, gru_layer, drop_layer,
                                                    input_list, train_list, test_list)
        red_player_embed = Add()(red_player_embeds)
        blue_player_embed = Add()(blue_player_embeds)
        y = Subtract()([red_player_embed, blue_player_embed])
        # bias=True for red/blue team difference.
        y_act = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='uniform')(y)
    elif params['idx'] == 2:
        # shape = (, 2 * params['n_hidden']) if bidirectional
        red_player_embeds = team_player_lstm_embed(train_data, test_data, match_hist_data, 'red',
                                                   mask_layer, gru_layer, drop_layer,
                                                   input_list, train_list, test_list)
        blue_player_embeds = team_player_lstm_embed(train_data, test_data, match_hist_data, 'blue',
                                                    mask_layer, gru_layer, drop_layer,
                                                    input_list, train_list, test_list)
        # output_shape: 5 * 2 * params['n_hidden']
        red_player_embed = Concatenate()(red_player_embeds)
        blue_player_embed = Concatenate()(blue_player_embeds)
        champ_embed = Embedding(input_dim=params['champion_num'], output_dim=2 * params['n_hidden'],
                                input_length=5, embeddings_initializer='uniform')
        champ_bias = Embedding(input_dim=params['champion_num'], output_dim=1,
                               input_length=5, embeddings_initializer='uniform')
        red_champ_embeds, red_champ_bias = team_champ_embed(champ_embed, champ_bias, train_data, test_data, 'red',
                                                            input_list, train_list, test_list)
        blue_champ_embeds, blue_champ_bias = team_champ_embed(champ_embed, champ_bias, train_data, test_data, 'blue',
                                                              input_list, train_list, test_list)
        red_champ_embed = Reshape((5 * 2 * params['n_hidden'],))(red_champ_embeds)
        blue_champ_embed = Reshape((5 * 2 * params['n_hidden'],))(blue_champ_embeds)
        red_player_champ_dot = Dot(axes=-1)([red_player_embed, red_champ_embed])
        blue_player_champ_dot = Dot(axes=-1)([blue_player_embed, blue_champ_embed])
        red_champ_bias_sum = Lambda(lambda x: K.sum(x, axis=1))(red_champ_bias)
        blue_champ_bias_sum = Lambda(lambda x: K.sum(x, axis=1))(blue_champ_bias)
        # output shape: 4
        y = Lambda(lambda x: K.concatenate([x[0], -x[1], x[2], -x[3]])) \
            ([red_player_champ_dot, blue_player_champ_dot, red_champ_bias_sum, blue_champ_bias_sum])
        # using following will cause AttributeError: 'Tensor' object has no attribute '_keras_history'
        # y = Concatenate()([red_player_champ_dot, K.tf.multiply(-1., blue_player_champ_dot),
        #                    red_champ_bias_sum, K.tf.multiply(-1., blue_champ_bias_sum)])
        # bias=True for red/blue team difference. all weights will be normalized relative to the bias's weight
        y_act = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='ones', trainable=False)(y)
    elif params['idx'] == 3:
        # set trainable to False as freeze layer.
        # see https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers
        champ_one_hot = Embedding(input_dim=params['champion_num'], output_dim=params['champion_num'], input_length=1,
                                  embeddings_initializer='identity', trainable=False)
        player_champ_matrix = Dense(params['n_hidden'], activation='relu', bias=True, kernel_initializer='uniform')
        red_player_champ_embeds = team_player_champ_lstm_embed(train_data, test_data, match_hist_data, 'red',
                                                               mask_layer, gru_layer, drop_layer,
                                                               champ_one_hot, player_champ_matrix,
                                                               input_list, train_list, test_list)
        blue_player_champ_embeds = team_player_champ_lstm_embed(train_data, test_data, match_hist_data, 'blue',
                                                                mask_layer, gru_layer, drop_layer,
                                                                champ_one_hot, player_champ_matrix,
                                                                input_list, train_list, test_list)
        red_player_champ_embed = merge(red_player_champ_embeds, mode='sum', output_shape=(params['n_hidden'],))
        blue_player_champ_embed = merge(blue_player_champ_embeds, mode='sum', output_shape=(params['n_hidden'],))
        y = subtract([red_player_champ_embed, blue_player_champ_embed])
        # bias=True for red/blue team difference.
        y_act = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='uniform')(y)
    elif params['idx'] == 4:
        # set trainable to False as freeze layer.
        # see https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers
        champ_one_hot = Embedding(input_dim=params['champion_num'], output_dim=params['champion_num'], input_length=1,
                                  embeddings_initializer='identity', trainable=False)
        player_champ_matrix = Dense(params['n_hidden'], activation='relu', use_bias=True, kernel_initializer='uniform')
        red_player_champ_embeds = team_player_champ_lstm_embed(train_data, test_data, match_hist_data, 'red',
                                                               mask_layer, gru_layer, drop_layer,
                                                               champ_one_hot, player_champ_matrix,
                                                               input_list, train_list, test_list)
        blue_player_champ_embeds = team_player_champ_lstm_embed(train_data, test_data, match_hist_data, 'blue',
                                                                mask_layer, gru_layer, drop_layer,
                                                                champ_one_hot, player_champ_matrix,
                                                                input_list, train_list, test_list)
        red_player_champ_embed = merge(red_player_champ_embeds, mode='sum', output_shape=(params['n_hidden'],))
        blue_player_champ_embed = merge(blue_player_champ_embeds, mode='sum', output_shape=(params['n_hidden'],))
        player_champ_embed_diff = subtract([red_player_champ_embed, blue_player_champ_embed])
        # bias=True for red/blue team difference.
        z = Dense(1, activation=None, bias=False, kernel_initializer='uniform')(player_champ_embed_diff)
        syng_mat = Dense(params['n_hidden'], activation=None, use_bias=False, kernel_initializer='uniform')
        oppo_mat = Dense(params['n_hidden'], activation=None, use_bias=False, kernel_initializer='uniform')
        syng_red = synergy(red_player_champ_embeds, syng_mat)
        syng_blue = synergy(blue_player_champ_embeds, syng_mat)
        oppo = opposition(red_player_champ_embeds, blue_player_champ_embeds, oppo_mat)
        y = merge([z, syng_red, K.tf.multiply(-1, syng_blue), oppo], mode='concat', output_shape=(4,))
        # bias=True for red/blue team difference. all weights will be normalized relative to the bias's weight
        y_act = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='ones', trainable=False)(y)
    elif params['idx'] == 5:
        # set trainable to False as freeze layer.
        # see https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers
        champ_one_hot = Embedding(input_dim=params['champion_num'], output_dim=params['champion_num'], input_length=1,
                                  embeddings_initializer='identity', trainable=False)
        red_player_champ_embed = team_player_champ_lstm_nn(train_data, test_data, match_hist_data, 'red',
                                                           mask_layer, gru_layer, drop_layer, champ_one_hot,
                                                           input_list, train_list, test_list)
        blue_player_champ_embed = team_player_champ_lstm_nn(train_data, test_data, match_hist_data, 'blue',
                                                            mask_layer, gru_layer, drop_layer, champ_one_hot,
                                                            input_list, train_list, test_list)
        y = merge([red_player_champ_embed, K.tf.multiply(-1, blue_player_champ_embed)],
                  mode='sum', output_shape=(params['champion_num'] * (params['n_hidden'] + 1),))
        # bias=True for red/blue team difference. all weights will be normalized relative to the bias's weight
        y_act = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='ones', trainable=False)(y)
    elif params['idx'] == 6:
        # set trainable to False as freeze layer.
        # see https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers
        champ_one_hot = Embedding(input_dim=params['champion_num'], output_dim=params['champion_num'], input_length=1,
                                  embeddings_initializer='identity', trainable=False)
        red_player_champ_embed = team_player_champ_lstm_nn(train_data, test_data, match_hist_data, 'red',
                                                           mask_layer, gru_layer, drop_layer, champ_one_hot,
                                                           input_list, train_list, test_list)
        blue_player_champ_embed = team_player_champ_lstm_nn(train_data, test_data, match_hist_data, 'blue',
                                                            mask_layer, gru_layer, drop_layer, champ_one_hot,
                                                            input_list, train_list, test_list)
        team_embed_matrix = Dense(params['n_hidden1'], activation='relu', use_bias=True, kernel_initializer='uniform')
        red_team_embed = team_embed_matrix(red_player_champ_embed)
        blue_team_embed = team_embed_matrix(blue_player_champ_embed)
        y = merge([red_team_embed, K.tf.multiply(-1, blue_team_embed)], mode='sum', output_shape=(params['n_hidden1'],))
        # bias=True for red/blue team difference. all weights will be normalized relative to the bias's weight
        y_act = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='ones', trainable=False)(y)

    objective = 'binary_crossentropy'
    metric = [acc]
    model = Model(inputs=input_list, outputs=y_act)
    model.compile(loss=objective, optimizer=Adam(lr=params['lr']), metrics=metric)
    return model, train_list, test_list


def run_model(x_train, x_test, x_valid, y_train, y_test, y_valid, match_hist_data):
    # x: raw data, X: transformed data
    model, X_train, X_test = create_model(x_train, x_test, match_hist_data)
    hist = model.fit(x=X_train, y=y_train, batch_size=params['batch_size'], verbose=2,
                     epochs=params['n_epochs'], validation_data=(X_test, y_test))
    y_score = model.predict(X_test, batch_size=params['batch_size'], verbose=0)
    y_pred = (np.ravel(y_score) > 0.5).astype('int32')
    return y_pred, hist.history


def train():
    with open(params['match_hist_path'], 'rb') as f:
        match_hist_data, _, _, _ = pickle.load(f)
    with open(params['match_path'], 'rb') as f:
        match_data = pickle.load(f)
        assert len(match_data.keys()) >= params['fold']
    # for each time series data point,
    # its dimension is the feature number (excluding champion id) + champion one hot encoding
    params['seq_feat_num'] = len(match_hist_data[0][0]) + params['champion_num'] - 1

    for fold in range(params['fold']):
        # original data format: match_num x each dict contains id of players and champions
        x_train, y_train = match_data[fold]['train'][0], match_data[fold]['train'][1]
        x_test, y_test = match_data[fold]['test'][0], match_data[fold]['test'][1]
        x_valid, y_valid = match_data[fold]['valid'][0], match_data[fold]['valid'][1]
        y_pred, hist = run_model(x_train, x_test, x_valid, y_train, y_test, y_valid, match_hist_data)
        res = evaluate(y_test, y_pred)


if __name__ == '__main__':
    with open('../input/lol_basic.pickle', 'rb') as f:
        # M: number of champions, N: number of players
        champion_id2idx_dict, M, summoner_id2idx_dict, N = pickle.load(f)

    params = {
              'champion_num': M,
              'match_hist_path': '../input/lol_lstm_match_hist_small.pickle',
              'match_path': '../input/lol_lstm_match_small.pickle',
              'fold': 1,
              'seq_max_len': 1000,
              'batch_size': 16,
              'lr': 0.001,
              'dropout': 0.0,
              'n_epochs': 500,
              'n_hidden': 8,
              # some models require two types of hidden units
              'n_hidden1': 8,
              'n_classes': 1,
              # 'bias': 1,
              # 'is_clf': 1,
              'idx': 2,
              # 1: player lstm average,
              # 2: player lstm - champion embedding dot product,
              # 3: player contextualized by champion one-hot 
              # 4: 3 + synergy + opposition
              # 5: fully-connected nn, each champion has a lstm block, then two teams sum
              # 6: fully-connected nn, each champion has a lstm block,
              # then form a team embedding, then two team embedding sum
              }
    train()
