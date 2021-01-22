
import player
import game
import neural_network
import mcts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import board as b
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from copy import deepcopy
import pandas as pd
import numpy as np

# model = keras.models.load_model('./best_keras_model.tf')



test = np.array([[ -1,  0,  1],
                [ 0,  1,  0],
                [ 0,  0, 0]])

board = b.Board()
board.board = test

player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type="w", temperature=1)
player2 = player.Zero_Player('o', 'Bot_ZERO', nn_type="w", temperature=1)

# player2.value_estimate = "nn"
# player2.keras_nn = model
player1.load_keras()
player2.load_keras()

self_play_game = game.Game(player1, player2, board, player2)
self_play_game.board = test
self_play_results = self_play_game.play(100)

augmented_self_play_results = neural_network.augment_data_set(self_play_results)

mcts.MCTS.get_tree_and_edges(reset=True)

mcts.MCTS.update_mcts_edges(augmented_self_play_results)

tree, edge_statistics, nodes = mcts.MCTS.get_tree_and_edges()
X_clean, Y_value, Y_policy = neural_network.update_nn_training_set(edge_statistics, nodes)
# master_df = neural_network.create_data_from_mcts(edge_statistics, nodes)
len(X_clean)