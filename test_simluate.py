import sys

new_path = [
 '/Users/yidinghou/.local/lib/python3.7/site-packages',
 '/Users/yidinghou/anaconda3/lib/python3.7/site-packages']

sys.path += new_path

import os
import numpy as np
import board
import pickle
import board as Board
import player
import game
from copy import deepcopy
from mcts import *

from neural_network import*

player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type="w", temperature=1)
player2 = player.Zero_Player('o', 'Bot_ZERO', nn_type="w", temperature=1)

brd = b.Board()
tree = MCTS()
PATH = {"path":[]}

players = [player1, player2]
test = np.array([[  1,   -1,  1],
				 [  0,   1,   0],
				 [  -1,   0,  -1]])
turn = 6
n_map = {-1:"o", 0: " ", 1: "x"}

str_state = ''.join([n_map[i] for i in test.reshape(9,)])
str_state

brd.board = test
simulate(players, turn, brd, tree.TREE, PATH, root = brd, n = 10)
tree.update_nodes()

player2.tree = tree.TREE
player1.tree = tree.TREE

print(player1.turn(brd))

