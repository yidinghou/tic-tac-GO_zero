import board as b
import numpy as np
import player
import keras
import neural_network

board = b.Board()
test = np.array([[ -1,  0,  -1],
                 [ 0,   -1, 1],
                 [1,   -1,  1]])

y = neural_network.CNN_Model()
c = neural_network.CNN_Model()

board.board	= test

player2 = player.Zero_Player('x', 'Bot_ZERO', nn_type="best", temperature=0)
player2.keras_nn = keras.models.load_model('./best_keras_model.tf')
# player2.value_estimate  = "nn"
print(player2.turn(board))


player2.value_estimate  = "nn"
print(player2.turn(board))