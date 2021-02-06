import sys

new_path = [
 '/Users/yidinghou/.local/lib/python3.7/site-packages',
 '/Users/yidinghou/anaconda3/lib/python3.7/site-packages']

sys.path += new_path
sys.setrecursionlimit(100000)

import os
import numpy as np
import board
import pickle
import board as Board
import player
import game
from copy import deepcopy
import mcts
from keras import regularizers
from neural_network import*

player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type="w", temperature=.3)
player2 = player.Zero_Player('o', 'Bot_ZERO', nn_type="w", temperature=.3)
players = [player1, player2]
tree = mcts.MCTS()

brd = Board.Board()

initial = np.array([[  0,   0,  0],
                 [  0,   0,   0],
                 [  0,   0,  0]])


import keras
from keras.layers import Input, Dense, Concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling1D, MaxPooling2D
from keras.optimizers import SGD
from keras import initializers
from keras.regularizers import l2

Input_1 = Input(shape=(3, 3, 1))

x1 = Conv2D(filters=4, kernel_size=(1, 3), activation='relu',
            kernel_regularizer=l2(0.0005),
            kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
            input_shape=(3, 3, 1))(Input_1)

x2 = Conv2D(filters=4, kernel_size=(3, 1), activation='relu',
            kernel_regularizer=l2(0.0005),
            kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
            input_shape=(3, 3, 1))(Input_1)

x3 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
            kernel_regularizer=l2(0.0005),
            kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
            input_shape=(3, 3, 1))(Input_1)

x1 = MaxPooling2D((3, 1))(x1)
x2 = MaxPooling2D((1, 3))(x2)
x3 = MaxPooling2D((1, 1))(x3)

x = Concatenate()([x1, x2, x3])
x = Flatten()(x)

value_head = Dense(10, activation='relu')(x)
value_head = Dense(1, activation='relu', name="V")(value_head)

policy_head = Dense(90, activation='relu')(x)
policy_head = Dense(10, activation='softmax', name="P", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(policy_head)

model = Model(inputs=Input_1, outputs=[value_head, policy_head])

# load old
model = keras.models.load_model("./best_keras_model.tf")
tree.load_tree_edges()

opt = SGD(lr=0.01, momentum=0.09)
model.compile(optimizer="adam",
              loss={"V": "mse", "P":"categorical_crossentropy"},
              loss_weights = [1,0],
              metrics=['acc'])


player1.keras_nn = model
player2.keras_nn = model


def keras_copy(model):
    model_copy = keras.models.clone_model(model)
    model_copy.compile(optimizer=opt,
                       loss={"V": "mse", "P": "categorical_crossentropy"},
                       loss_weights=[1, 0],
                       metrics=['acc'])
    model_copy.set_weights(model.get_weights())

    return model_copy


for i in range(2):
    prior_model = keras_copy(model)

    brd.board = initial
    PATH = {"path": []}
    player1.value_estimate = ""
    player2.value_estimate = ""
    player1.temperature = .5
    player2.temperature = .5

    self_play_game = game.Game(player1, player2, tree)
    self_play_game.board = brd
    self_play_results = self_play_game.play(100)
    tree.save_tree_edges()
    print("saved")
    train_data = update_nn_training_set(tree.EDGES, tree.TREE)

    X_clean = train_data[0]
    Y_value = train_data[1]
    # values are between 0 and 1, multiply by 2 to get range(0,2)
    targets = (Y_value * 2).round()
    Y_value_one_hot = np.eye(3)[targets.astype(int)]

    Y_policy = train_data[2]

    X_final = np.stack([X_clean], axis=-1)
    model.fit(X_final, [Y_value, Y_policy], epochs=100, verbose=1)

    N_games = 100
    global_step = 50000

    player1.value_estimate ="nn"
    player1.temperature = 0
    player2.temperature = 0
    player2.value_estimate="nn"

    player1.keras_nn = prior_model
    player2.keras_nn = model

    z_vs_r_game = game.Game(player1, player2, tree)
    w1, w2 = z_vs_r_game.play_symmetric(N_games)
    print('{} vs {} summary:'.format(player1.name, player2.name))
    print('wins={}, draws={}, losses={}'.format(w1, N_games - w1 - w2, w2))

    if w2 >= (w1 * 1.05):
        print("saving new best model")
        best_model = keras_copy(model)

        player1.keras_nn = best_model
        player2.keras_nn = best_model
        best_model.save("./best_keras_model.tf")
    else:
        model = prior_model
        player2.keras_nn = prior_model


tree.save_tree_edges()