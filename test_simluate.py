import os
import numpy as np
import board
import pickle
import board as Board
import player
import game
from copy import deepcopy

from neural_network import*

player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type="w", temperature=1)
player2 = player.Zero_Player('o', 'Bot_ZERO', nn_type="w", temperature=1)


class Node():
	def __init__(self, id_):
		self.id = id_
		self.parent = []
		self.children = []
		self.eval = False
		self.value = 0
		self.W = 0
		self.D = 0
		self.L = 0
		self.P = 0
		self.Q = 0
		self.N = 0
		self.Turn = 0

		self.board_arr = Board.str2arr(id_)

		turns = len(id_.replace(" ", ""))
		idx = (turns) % 2
		self.turn = [1, -1][idx]

	def is_leaf(self):
		return len(self.children) == 0

	def is_root(self):
		return len(self.parent) == 0


def eval_to_leaf(parent):
	while parent.eval == False:
		middle = [node for node in parent.children if not node.eval]
		for mid_node in middle:
			eval_to_leaf(mid_node)

		all_eval = len(middle) == 0
		if all_eval:
			if parent.turn == 1:
				value_funt = max
			else:
				value_funt = min
			values = [node.value for node in parent.children]
			parent.value = value_funt(values)
			parent.eval = True


TREE = {}


def get_node_id(new_node_id, TREE):
	board_arr = Board.str2arr(new_node_id)
	sym_brd = create_symmetry(board_arr)
	sym_id = [Board.arr2str(brd) for brd in sym_brd]

	intersection = list(sym_id & TREE.keys())

	return intersection


def create_symmetry(board_arr):
	rot_brd = [board_arr]
	for i in range(1, 4):
		rot = np.rot90(board_arr, k=i)
		rot_brd.append(rot)

	flip_brd = [np.flip(brd) for brd in rot_brd]

	sym_brd = rot_brd + flip_brd

	return sym_brd


def simulate(players, board, TREE, EDGES, turn):
	player_idx = turn % 2
	opp_idx = (turn + 1) % 2
	curr_player = players[player_idx]
	opp_player = players[opp_idx]

	possible_moves = np.where(board.board.ravel() == 0)[0]
	curr_state = Board.arr2str(board.board)

	node_id = get_node_id(curr_state, TREE)
	if len(node_id) == 0:
		parent_node = Node(curr_state)
		add_node(parent_node, TREE)
	else:
		parent_node = TREE[node_id[0]]

	parent_node.Turn = curr_player.type

	for move in possible_moves:
		board_copy = deepcopy(board)
		row, col = divmod(move, 3)
		board_copy.add_move(curr_player.type, row, col)
		winner = Board.Board.winner(board_copy.board)
		next_state = Board.arr2str(board_copy.board)

		node_id = get_node_id(next_state, TREE)
		if len(node_id) == 0:
			child_node = Node(next_state)
			add_node(child_node, TREE)
		else:
			child_node = TREE[node_id[0]]

		edge_id = parent_node.id + "2" + child_node.id

		if edge_id in EDGES.keys():
			EDGES[edge_id] += 1
		else:
			EDGES[edge_id] = 1

		child_node.Turn = opp_player.type
		child_node.parent.append(parent_node)
		parent_node.children.append(child_node)

		if winner != 0:
			if winner == 1:
				child_node.W += 1
				parent_node.W += 1
			elif winner == -1:
				child_node.L += 1
				parent_node.L += 1
		else:
			if not board_copy.full():
				simulate(players, board_copy, TREE, EDGES, turn + 1)


bod = b.Board()
TREE = {}
EDGES = {}

test = np.array([[  1,   -1,  1],
                 [  1,   1,   -1],
                 [  0,   0,  -1]])

n_map = {-1:"o", 0: " ", 1: "x"}

str_state = ''.join([n_map[i] for i in test.reshape(9,)])
str_state

bod.board = test

players = [player1, player2]
turn = 7

simulate(players, bod, TREE, EDGES,turn)