__author__ = 'Florin Bora'

import board as b

import pandas as pd
import os
import numpy as np
import keras
from keras.layers import Input, Dense, Concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling1D, MaxPooling2D
from keras.optimizers import SGD
from keras import initializers
from keras.regularizers import l2

def augment_data_set(data):
    augmented_data = data.copy()
    # rotations
    for game in reversed(augmented_data):
        for k in range(1, 4):
            augmented_data += [[[np.rot90(state) for state in game[0]], game[1]]]
    # inversions
    for game in reversed(augmented_data):
        augmented_data += [[[np.transpose(state) for state in game[0]], game[1]]]
    # ignore color reflections
    return augmented_data


def create_data_from_mcts(edge_statistics):
    board = b.Board()
    X = []
    Y = []
    N = []
    Y_move = []

    X_init_state = []
    X_final_state = []

    keys = list(edge_statistics.keys())
    for key in keys:
        state = edge_statistics[key]
        initial_state, final_state = key.split("2")
        X_init_state.append(initial_state)
        X_final_state.append(final_state)

        initial_arr = board.str2arr(initial_state)
        final_arr = board.str2arr(final_state)
        move = final_arr - initial_arr
        p_type = move.sum()
        move = move * p_type
        move = move.reshape(-1, 9)[0]
        move = np.where(move == 1)[0][0]

        Y.append(state["Q"])
        N.append(state["N"])
        Y_move.append(move)

    X = np.array(X)
    Y = np.array(Y)
    Y_move = np.array(Y_move)

    df = pd.DataFrame({
        'init_state': X_init_state,
        'final_state': X_final_state,
        'move': Y_move,
        'value': Y,
        'N': N
    })

    df = df[df["N"] > 0]
    values_df = df[["final_state", "value", "N"]]
    moves_df = df[["init_state", "move"]]
    master_df = values_df.merge(moves_df, left_on="final_state", right_on="init_state", how="left")
    master_df["move"].fillna(9, inplace=True)

    state_freq = master_df["final_state"].value_counts()
    inv_state_freq = sum(state_freq) / state_freq
    inv_state_freq.name = "inv_freq"
    inv_state_freq
    master_df = master_df.merge(inv_state_freq, left_on="final_state", right_index=True)

    return (master_df)


def update_nn_training_set(edge_statistics, n=10000):
    master_df = create_data_from_mcts(edge_statistics)
    data_sample = master_df.sample(n=n, weights='inv_freq', replace=True)
    board = b.Board()

    X = np.array(data_sample["final_state"])
    X = np.array([board.str2arr(state) for state in X])

    Y = np.array(data_sample["value"])
    # convert policy move index to one hot array
    Y_policy = np.eye(10)[np.array(data_sample["move"].values).astype(int)]

    nb_classes = 3
    targets = (Y+1)
    one_hot_targets = np.eye(nb_classes)[targets.astype(int)]

    return X, one_hot_targets, Y_policy


def train_nn(model, edge_statistics, iterations=10):

    verbose = 0
    for i in range(iterations):
        train_data = update_nn_training_set(edge_statistics,  n=10000)
        X_clean = train_data[0]
        Y_value = train_data[1]
        Y_policy = train_data[2]

        if (i+1) ==10:
            verbose = 1

        X_final = np.stack([X_clean], axis=-1)
        model.fit(X_final, [Y_value, Y_policy], epochs=10, verbose=verbose)

    return model


def CNN_Model():
    Input_1 = Input(shape=(3, 3, 1))

    x1 = Conv2D(filters=6, kernel_size=(1, 3), activation='relu',
                kernel_regularizer=l2(0.0005),
                kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
                input_shape=(3, 3, 1))(Input_1)

    x2 = Conv2D(filters=6, kernel_size=(3, 1), activation='relu',
                kernel_regularizer=l2(0.0005),
                kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
                input_shape=(3, 3, 1))(Input_1)

    x3 = Conv2D(filters=10, kernel_size=(3, 3), activation='relu',
                kernel_regularizer=l2(0.0005),
                kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
                input_shape=(3, 3, 1))(Input_1)

    x1 = MaxPooling2D((3, 1))(x1)
    x2 = MaxPooling2D((1, 3))(x2)
    x3 = MaxPooling2D((1, 1))(x3)

    x = Concatenate()([x1, x2, x3])
    x = Flatten()(x)

    value_head = Dense(10, activation='relu')(x)
    value_head = Dense(3, activation='softmax', name="V")(value_head)

    policy_head = Dense(90, activation='relu')(x)
    policy_head = Dense(10, activation='softmax', name="P")(policy_head)

    model = Model(inputs=Input_1, outputs=[value_head, policy_head])
    opt = SGD(lr=0.1, momentum=0.09)
    model.compile(optimizer=opt,
                         loss={"P": 'categorical_crossentropy', "V": "categorical_crossentropy"},
                         metrics=['acc'])
    return model

class nn_predictor():
    BEST = None
    LAST = None
    CHECK_POINTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn_check_pts')
    CHECK_POINTS_NAME = os.path.join(CHECK_POINTS_DIR, 'nn')
    META = os.path.join(CHECK_POINTS_DIR, 'nn.meta')

    def __init__(self, nn_type):
        if nn_type in ['best', 'last']:
            if nn_predictor.LAST is None or nn_predictor.BEST is None:
                # if no model was ever constructed
                self.model = CNN_Model()
                with tf.Session(graph=self.model.graph) as sess:
                    sess.run(self.model.init_op)
                    saver = tf.train.Saver(tf.global_variables())
                    saver.export_meta_graph(nn_predictor.META)
                    init_model = saver.save(sess, nn_predictor.CHECK_POINTS_NAME,
                        global_step=self.model.global_step, write_meta_graph=False)
                nn_predictor.LAST = init_model
                nn_predictor.BEST = init_model
            check_point = nn_predictor.LAST if nn_type == 'last' else nn_predictor.BEST
        else:
            check_point = nn_type

        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(nn_predictor.META)
        saver.restore(self.sess, check_point)
        self.X_tf = self.sess.graph.get_tensor_by_name('X:0')
        self.y_tf = self.sess.graph.get_tensor_by_name('activation3:0')

    def predict(self, input):
        input_np = input.reshape(-1, 9)
        output_np = self.sess.run(self.y_tf, feed_dict={self.X_tf: input_np})
        output_np = output_np.reshape(12)
        return output_np[:3], output_np[3:]

    @classmethod
    def reset_nn_check_pts(cls):
        if not os.path.isdir(cls.CHECK_POINTS_DIR):
            os.mkdir(cls.CHECK_POINTS_DIR)
        for file in os.listdir(cls.CHECK_POINTS_DIR):
            os.remove(os.path.join(cls.CHECK_POINTS_DIR, file))

def main():
    print('')

if __name__ == '__main__':
    main()
