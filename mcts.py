import os
import numpy as np
import board
import pickle
import board as Board
from copy import deepcopy

class MCTS():
    def __init__(self):
        self.MCTS_DIR = os.path.join('/Users/yidinghou/Desktop/Projects/tic-tac-GO_zero', 'mcts')
        self.PUCT_CONSTANT = 10.0
        self.TREE_FILE = 'tree.pkl'
        self.NODES_FILE = 'nodes.pkl'
        self.TREE_PATH = os.path.join(self.MCTS_DIR, self.TREE_FILE)
        self.NODES_PATH = os.path.join(self.MCTS_DIR, self.NODES_FILE)
        self.TREE = {}


    def get_tree_and_edges(self, reset=False):
        if not os.path.isdir(self.MCTS_DIR):
            os.mkdir(self.MCTS_DIR)

        if reset:
            if os.path.isfile(self.TREE_PATH):
                os.remove(self.TREE_PATH)

    def save_tree_edges(self):
        with open(self.TREE_PATH, 'wb') as t:
            pickle.dump(self.TREE, t, pickle.HIGHEST_PROTOCOL)


    def load_tree_edges(self):
        with open(self.TREE_PATH, 'rb') as t:
            tree = pickle.load(t)
            self.TREE = tree


    def update_nodes(self):
        for nodes in self.TREE.values():
            nodes.Q = (nodes.W - nodes.L)/nodes.N
            nodes.Q = nodes.Q/2 + 0.5


def PUCT_function(PUCT_CONSTANT, bool, node):
    if bool == -1:
        Q = 1 - node.Q
    else:
        Q = node.Q

    puct = Q + PUCT_CONSTANT * node.P * np.sqrt(node.N) / (1 + node.N)
    return puct


def print_edges(edges):
    for k in edges.keys():
        if edges[k]['N'] == 0:
            print('|{}|'.format(k), edges[k])


def is_node_in_tree(new_node_id, TREE):
    board_arr = Board.str2arr(new_node_id)
    sym_brd = create_symmetry(board_arr)
    sym_id = [Board.arr2str(brd) for brd in sym_brd]

    intersection = list(sym_id & TREE.keys())

    if len(intersection) == 0:
        return False, new_node_id

    return True, intersection[0]


def create_symmetry(board_arr):
    rot_brd = [board_arr]
    for i in range(1, 4):
        rot = np.rot90(board_arr, k=i)
        rot_brd.append(rot)

    flip_brd = [np.flip(brd) for brd in rot_brd]

    sym_brd = rot_brd + flip_brd

    return sym_brd


def get_node(node_id, TREE):
    node_in_tree, node_id = is_node_in_tree(node_id, TREE)
    if not node_in_tree:
        new_node = Node(node_id)
        TREE[node_id] = new_node
        return new_node
    else:
        return TREE[node_id]


class Node():
    def __init__(self, id_):
        self.id = id_
        self.parents = []
        self.children = []
        self.value_eval = False
        self.N_eval = False

        self.value = 0
        self.edges = []

        self.W = 0
        self.D = 0
        self.L = 0
        self.P = 0
        self.Q = 0
        self.N = 0
        self.PUCT=0
        self.Turn = 0

        self.board_arr = Board.str2arr(id_)

        turns = len(id_.replace(" ", ""))
        idx = (turns) % 2
        self.turn = [1, -1][idx]

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return len(self.parents) == 0


def simulate(players, turn, board, TREE, PATH, root, n=1):
    simulations = 0
    player_idx = turn % 2
    opp_idx = (turn+1) % 2
    curr_player = players[player_idx]
    opp_player = players[opp_idx]

    curr_state = Board.arr2str(board.board)
    parent_node = get_node(curr_state, TREE)
    parent_node.Turn = curr_player.type
    winner = Board.winner(board.board)
    game_over = board.full() or (winner!=0)
    if game_over:
        if winner == 1:
            parent_node.W += 1
        elif winner == -1:
            parent_node.L += 1
        return

    possible_moves = np.where(board.board.ravel() == 0)[0]

    while simulations < n:
        simulations += 1
        board_copy = deepcopy(board)

        move = np.random.choice(possible_moves, 1)
        row, col = divmod(move, 3)
        board_copy.add_move(curr_player.type, row, col)
        next_state = Board.arr2str(board_copy.board)
        PATH["path"].append(next_state)

        child_node = get_node(next_state, TREE)

        child_node.Turn = opp_player.type
        if parent_node not in child_node.parents:
            child_node.parents.append(parent_node)
        if child_node not in parent_node.children:
            parent_node.children.append(child_node)

        winner = Board.winner(board_copy.board)
        game_over = board_copy.full() or (winner!=0)

        if not game_over:
            simulate(players, turn + 1, board_copy, TREE, PATH, root, n=1)
        else:
            update_eval_tree(TREE, PATH, root, winner)
            PATH["path"]=[]


def update_eval_tree(TREE, PATH, root, winner):
    root_node = TREE[Board.arr2str(root.board)]
    root_node.N+=1
    if winner ==1:
        root_node.W+=1
    elif winner ==-1:
        root_node.L+=1
    else:
        root_node.D+=1

    for node_id in PATH["path"]:
        TREE[node_id].N+=1
        if winner ==1:
            TREE[node_id].W+=1
        elif winner ==-1:
            TREE[node_id].L+=1
        else:
            TREE[node_id].D+=1

