import board as b
import numpy as np
import player
import keras
import neural_network
import mcts
import game

mcts.MCTS.get_tree_and_edges(reset=True)
# neural_network.nn_predictor.reset_nn_check_pts()
# nn_training_set = None

iterations = 50
player1 = player.Zero_Player('x', 'Bot_ONE', nn_type='best', temperature=1)
player2 = player.Zero_Player('o', 'Bot_ONE', nn_type='best', temperature=1)

player1.load_keras()
player2.load_keras()


# player1.value_estimate = "nn"
# player1.value_estimate = "nn"
self_play_game = game.Game(player1, player2, b.Board(), player1)
self_play_results = self_play_game.play(5000)

augmented_self_play_results = neural_network.augment_data_set(self_play_results)

mcts.MCTS.update_mcts_edges(augmented_self_play_results)


# import keras
# from keras.layers import Input, Dense, Concatenate
# from keras.models import Sequential, Model
# from keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling1D, MaxPooling2D
# from keras.optimizers import SGD
# from keras import initializers
# from keras.regularizers import l2
#
#
# tree, edge_statistics, nodes = mcts.MCTS.get_tree_and_edges()
# X_clean, Y_value, Y_policy = neural_network.update_nn_training_set(edge_statistics, nodes)
# master_df = neural_network.create_data_from_mcts(edge_statistics, nodes)
# len(X_clean)
#
#
#
# Input_1 = Input(shape=(3, 3, 1))
#
# x1 = Conv2D(filters=6, kernel_size=(1, 3), activation='relu',
#             kernel_regularizer=l2(0.0005),
#             kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
#             input_shape=(3, 3, 1))(Input_1)
#
# x2 = Conv2D(filters=6, kernel_size=(3, 1), activation='relu',
#             kernel_regularizer=l2(0.0005),
#             kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
#             input_shape=(3, 3, 1))(Input_1)
#
# x3 = Conv2D(filters=10, kernel_size=(3, 3), activation='relu',
#             kernel_regularizer=l2(0.0005),
#             kernel_initializer=initializers.RandomNormal(stddev=0.1, mean=0),
#             input_shape=(3, 3, 1))(Input_1)
#
# x1 = MaxPooling2D((3, 1))(x1)
# x2 = MaxPooling2D((1, 3))(x2)
# x3 = MaxPooling2D((1, 1))(x3)
#
# x = Concatenate()([x1, x2, x3])
# x = Flatten()(x)
#
# # value_head = Dense(10, activation='relu')(x)
# value_head = Dense(3, activation='softmax', name="V")(x)
#
# # policy_head = Dense(90, activation='relu')(x)
# policy_head = Dense(10, activation='softmax', name="P")(x)
#
# model = Model(inputs=Input_1, outputs=[value_head, policy_head])
# opt = SGD(lr=0.01, momentum=0.009)
# model.compile(optimizer="adam",
#                      loss={"P": 'mae', "V": "categorical_crossentropy"},
#                      metrics=['acc'])
#
# X_final = np.stack([X_clean], axis=-1)
# model.fit(X_final, [Y_value, Y_policy], epochs=100, verbose=1)
# model.save("./best_keras_model.tf")
