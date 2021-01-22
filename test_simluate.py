import os
import numpy as np
import board
import pickle
import board as b
import player
import game
from copy import deepcopy

from neural_network import*


class MCTS():
	def __init__(self):
		self.MCTS_DIR = os.path.join('/Users/yidinghou/Desktop/Projects/tic-tac-GO_zero', 'mcts')
		self.PUCT_CONSTANT = 10.0
		self.TREE_FILE = 'tree.pkl'
		self.EDGES_FILE = 'edges.pkl'
		self.NODES_FILE = 'nodes.pkl'
		self.TREE_PATH = os.path.join(self.MCTS_DIR, self.TREE_FILE)
		self.EDGES_PATH = os.path.join(self.MCTS_DIR, self.EDGES_FILE)
		self.NODES_PATH = os.path.join(self.MCTS_DIR, self.NODES_FILE)

		self.WIN2DICT_MAP = {-1: 'L', 0: 'D', 1: 'W'}

	def get_tree_and_edges(self, reset=False):
		if not os.path.isdir(self.MCTS_DIR):
			os.mkdir(self.MCTS_DIR)

		if reset:
			if os.path.isfile(self.TREE_PATH):
				os.remove(self.TREE_PATH)
			if os.path.isfile(self.EDGES_PATH):
				os.remove(self.EDGES_PATH)
			if os.path.isfile(self.NODES_PATH):
				os.remove(self.NODES_PATH)

		if not (os.path.isfile(self.TREE_PATH) or os.path.isfile(self.EDGES_PATH) or os.path.isfile(self.NODES_PATH)):
			tree, edges, nodes = board.Board.generate_state_space()
			self.save_tree_edges(tree, edges, nodes)
		else:
			tree, edges, nodes = self.load_tree_edges()
		return tree, edges, nodes

	def save_tree_edges(self, tree, edges, nodes):
		with open(self.TREE_PATH, 'wb') as t:
			pickle.dump(tree, t, pickle.HIGHEST_PROTOCOL)
		with open(self.EDGES_PATH, 'wb') as e:
			pickle.dump(edges, e, pickle.HIGHEST_PROTOCOL)
		with open(self.NODES_PATH, 'wb') as n:
			pickle.dump(nodes, n, pickle.HIGHEST_PROTOCOL)

	def load_tree_edges(self):
		with open(self.TREE_PATH, 'rb') as t:
			tree = pickle.load(t)
		with open(self.EDGES_PATH, 'rb') as e:
			edges = pickle.load(e)
		with open(self.NODES_PATH, 'rb') as n:
			nodes = pickle.load(n)
		return tree, edges, nodes

	def update_mcts_edges(self, new_games):
		tree, edges, NODES = self.get_tree_and_edges()
		for game in new_games:
			win = game[1]
			node = board.Board.arr2str(game[0][0])
			NODES[node]['N'] += 1
			NODES[node][self.WIN2DICT_MAP[win]] += 1
			NODES[node]['Q'] = NODES[node]['W'] / NODES[node]['N']

			for i in range(len(game[0]) - 1):
				initial = game[0][i]
				final = game[0][i + 1]

				edge = board.Board.arr2str(initial) + '2' + board.Board.arr2str(final)
				edges[edge]['N'] += 1

				node = board.Board.arr2str(final)
				NODES[node]['N'] += 1
				NODES[node][self.WIN2DICT_MAP[win]] += 1
				NODES[node]['Q'] = NODES[node]['W'] / NODES[node]['N']

	def PUCT_function(self, bool, N, edge):
		if bool == -1:
			Q = 1 - edge['Q']
		else:
			Q = edge['Q']

		puct = Q + self.PUCT_CONSTANT * edge['P'] * np.sqrt(N) / (1 + edge['N'])
		return puct



curr_mcts = MCTS()
curr_mcts.get_tree_and_edges(reset=True)
tree, edge_statistics, nodes = curr_mcts.get_tree_and_edges()


def simulate(players, board, nodes, edges, turn):
	player_idx = turn % 2
	curr_player = players[player_idx]
	possible_moves = np.where(board.board.ravel() == 0)[0]
	curr_state = b.Board.arr2str(board.board)

	for move in possible_moves:
		board_copy = deepcopy(board)
		row, col = divmod(move, 3)
		board_copy.add_move(curr_player.type, row, col)
		winner = b.Board.winner(board_copy.board)
		next_state = b.Board.arr2str(board_copy.board)
		node_next = nodes[next_state]

		edges[curr_state + "2" + next_state]["N"] += 1

		if winner != 0:
			if winner == 1:
				node_next["W"] += 1
			elif winner == -1:
				node_next["L"] += 1

			node_next["N"] += 1
			node_next["Q"] = (node_next["W"] - node_next["L"]) / node_next["N"]
		else:
			if board_copy.full():
				node_next["N"] += 1
				node_next["Q"] = (node_next["W"] - node_next["L"]) / node_next["N"]
			else:
				simulate(players, board_copy, nodes, edges, turn + 1)


player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type="w", temperature=1)
player2 = player.Zero_Player('o', 'Bot_ZERO', nn_type="w", temperature=1)

bod = b.Board()

test = np.array([[  1,   0,  -1],
                 [ -1,  -1,  1],
                 [  0,   1,  1]])
bod.board = test

players = [player1, player2]
turn = 7

simulate(players, bod, nodes, edge_statistics, turn)

value_df = create_values_df(nodes)
moves_df = create_policy_df(edge_statistics)


class Node():
	def __init__(self, id_):
		self.id = id_
		self.parent = []
		self.children = []
		self.eval = False
		self.value = 0
		self.N = 0

		bod = b.Board()
		self.board_arr = bod.str2arr(id_)

		turns = len(id_.replace(" ", ""))
		idx = (turns) % 2
		self.turn = [1, -1][idx]

	def is_leaf(self):
		return len(self.children) == 0

	def is_root(self):
		return len(self.parent) == 0


node_tree_dict = {}

for idx, value in moves_df.iterrows():
	parent_id = value["init_state"]
	child_id = value["final_state"]

	if not parent_id in node_tree_dict.keys():
		parent_node = Node(parent_id)
		node_tree_dict[parent_id] = parent_node
	else:
		parent_node = node_tree_dict[parent_id]

	if not child_id in node_tree_dict.keys():
		child_node = Node(child_id)
		node_tree_dict[child_id] = child_node
	else:
		child_node = node_tree_dict[child_id]

	parent_node.children.append(child_node)
	child_node.parent.append(parent_node)

terminal = [x for x in node_tree_dict.values() if x.is_leaf()]
for node in terminal:
	board_arr = bod.str2arr(node.id)
	node.board_arr = board_arr
	winner = b.Board.winner(board_arr)
	node.value = winner
	node.N += 1
	node.eval = True

root = [x for x in node_tree_dict.values() if x.is_root()][0]


def eval_to_leaf(parent):
	while parent.eval == False:
		middle = [node for node in parent.children if not node.eval]
		for mid_node in middle:
			eval_to_leaf(mid_node)

		all_eval = len(middle) == 0
		if all_eval:
			values = [node.value * parent.turn for node in parent.children]
			parent.value = max(values)* parent.turn
			parent.eval = True

eval_to_leaf(root)