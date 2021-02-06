import os
import numpy as np
import board
import pickle
import board as Board
from copy import deepcopy
import player

class MCTS():
    def __init__(self):
        self.MCTS_DIR = os.path.join('/Users/yidinghou/Desktop/Projects/tic-tac-GO_zero', 'mcts')
        self.PUCT_CONSTANT = 10.0
        self.TREE_FILE = 'tree.pkl'
        self.NODES_FILE = 'nodes.pkl'
        self.TREE_PATH = os.path.join(self.MCTS_DIR, self.TREE_FILE)
        self.NODES_PATH = os.path.join(self.MCTS_DIR, self.NODES_FILE)
        self.TREE = {}
        self.EDGES = {}


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
        for nodes in [n for n in self.TREE.values() if n.N > 0]:
            nodes.Q = (nodes.W - nodes.L)/nodes.N
            nodes.Q = nodes.Q/2 + 0.5


def PUCT_function(PUCT_CONSTANT, bool, node):
    if bool == -1:
        Q = 1 - node.Q
    else:
        Q = node.Q

    puct = Q + PUCT_CONSTANT * np.sqrt(node.N) / (1 + node.N)
    return puct


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

    flip_brd = [np.flip(brd, axis=1) for brd in rot_brd]

    sym_brd = rot_brd + flip_brd

    return sym_brd


def get_node(node_id, TREE):
    if not node_id in TREE.keys():
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
        self.Q = 0.5
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


def simulate(players, turn, board, tree, PATH, root, n=1):
    TREE = tree.TREE
    EDGES = tree.EDGES

    simulations = 0
    player_idx = turn % 2
    opp_idx = (turn+1) % 2
    curr_player = players[player_idx]
    opp_player = players[opp_idx]

    curr_brd_sym = create_symmetry(board.board)
    curr_state_sym = [Board.arr2str(brd) for brd in curr_brd_sym]
    parent_node_sym =[get_node(node_id, TREE) for node_id in curr_state_sym]
    parent_node = parent_node_sym[0]

    parent_node.Turn = curr_player.type
    winner = Board.winner(board.board)
    game_over = board.full() or (winner!=0)

    if game_over:
        for sym in parent_node_sym:
            sym.N +=1
        if winner == 1:
            for sym in parent_node_sym:
                sym.N += 1
        elif winner == -1:
            for sym in parent_node_sym:
                sym.N += 1
        return

    while simulations < n:
        simulations += 1
        board_copy = deepcopy(board)

        if curr_player.name == "Human":
            bot_player = player.Zero_Player('o', 'Bot_ZERO', nn_type="w", temperature=.3)
            bot_player.value_estimate = "nn"
            bot_player.tree = tree
            bot_player.keras_nn = curr_player.keras_nn
            bot_player.type = curr_player.type
            _, row, col = bot_player.turn(board)
        else:
            _, row, col = curr_player.turn(board)


        board_copy.add_move(curr_player.type, row, col)
        next_state = Board.arr2str(board_copy.board)

        next_brd_sym = create_symmetry(board_copy.board)
        next_state_sym = [Board.arr2str(brd) for brd in next_brd_sym]
        child_node_sym = [get_node(node_id, TREE) for node_id in next_state_sym]

        PATH["path"].append(next_state)
        #adding in symmetries
        for i in range(0,8):
            curr_state = curr_state_sym[i]
            next_state = next_state_sym[i]
            edge_id = curr_state+"2"+next_state
            if not edge_id in EDGES.keys():
                EDGES[edge_id]=1
            else:
                EDGES[edge_id]+=1

            child_node = child_node_sym[i]
            parent_node = parent_node_sym[i]
            child_node.Turn = opp_player.type
            if parent_node not in child_node.parents:
                child_node.parents.append(parent_node)
            if child_node not in parent_node.children:
                parent_node.children.append(child_node)

        winner = Board.winner(board_copy.board)
        game_over = board_copy.full() or (winner!=0)

        if not game_over:
            simulate(players, turn + 1, board_copy, tree, PATH, root, n=1)
        else:
            update_eval_tree(tree, PATH, root, winner)
            tree.update_nodes()
            PATH["path"]=[]

    end = 1


def update_eval_tree(tree, PATH, root, winner):
    TREE = tree.TREE
    root_brd_sym = create_symmetry(root.board)
    root_state_sym = [Board.arr2str(brd) for brd in root_brd_sym]
    root_node_sym = [get_node(node_id, TREE) for node_id in root_state_sym]
    for node in root_node_sym:
        node.N+=1

    if winner ==1:
        for node in root_node_sym:
            node.W+=1
    elif winner ==-1:
        for node in root_node_sym:
            node.L+=1
    else:
        for node in root_node_sym:
            node.D+=1

    for node_id in PATH["path"]:
        brd_sym = create_symmetry(Board.str2arr(node_id))
        state_sym = [Board.arr2str(brd) for brd in brd_sym]
        node_sym = [get_node(node_id, TREE) for node_id in state_sym]

        for node in node_sym:
            node.N+=1
            if winner ==1:
                node.W+=1
            elif winner ==-1:
                node.L+=1
            else:
                node.D+=1
