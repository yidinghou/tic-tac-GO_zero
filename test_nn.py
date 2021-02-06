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
import mcts
import matplotlib.pyplot as plt

from neural_network import*

player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type="w", temperature=1)
player2 = player.Zero_Player('o', 'Bot_ZERO', nn_type="w", temperature=1)
players = [player1, player2]
tree = mcts.MCTS()
# tree.load_tree_edges()
brd = Board.Board()

test = np.array([[  1,   0,  0],
                 [  0,   -1,   0],
                 [  0,   0,  1]])

player1.tree = tree
player2.tree = tree

brd.board = test
# PATH = {"path":[]}
#
#
self_play_game = game.Game(player1, player2, tree)
self_play_game.board = brd
self_play_results = self_play_game.play(100)

test = np.array([[  1,   0,  0],
                 [  0,   -1,   0],
                 [  0,   0,  1]])
brd.board = test
player2.turn(brd)