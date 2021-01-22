__author__='Florin Bora'

import os
import numpy as np
import board
import pickle

class MCTS():

    MCTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mcts')
    PUCT_CONSTANT = 10.0
    TREE_FILE = 'tree.pkl'
    EDGES_FILE = 'edges.pkl'
    NODES_FILE = 'nodes.pkl'
    TREE_PATH = os.path.join(MCTS_DIR, TREE_FILE)
    EDGES_PATH = os.path.join(MCTS_DIR, EDGES_FILE)
    NODES_PATH = os.path.join(MCTS_DIR, NODES_FILE)

    WIN2DICT_MAP = {-1: 'L', 0: 'D', 1: 'W'}

    def __init__(self):
        pass


    @classmethod
    def get_tree_and_edges(cls, reset=False):
        if not os.path.isdir(cls.MCTS_DIR):
            os.mkdir(cls.MCTS_DIR)

        if reset:
            if os.path.isfile(cls.TREE_PATH):
                os.remove(cls.TREE_PATH)
            if os.path.isfile(cls.EDGES_PATH):
                os.remove(cls.EDGES_PATH)
            if os.path.isfile(cls.NODES_PATH):
                os.remove(cls.NODES_PATH)

        if not (os.path.isfile(cls.TREE_PATH) or os.path.isfile(cls.EDGES_PATH) or os.path.isfile(cls.NODES_PATH)):
            tree, edges, nodes = board.Board.generate_state_space()
            cls.save_tree_edges(tree, edges, nodes)
        else:
            tree, edges, nodes = cls.load_tree_edges()
        return tree, edges, nodes


    @classmethod
    def save_tree_edges(cls, tree, edges, nodes):
        with open(cls.TREE_PATH, 'wb') as t:
            pickle.dump(tree, t, pickle.HIGHEST_PROTOCOL)
        with open(cls.EDGES_PATH, 'wb') as e:
            pickle.dump(edges, e, pickle.HIGHEST_PROTOCOL)
        with open(cls.NODES_PATH, 'wb') as n:
            pickle.dump(nodes, n, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_tree_edges(cls):
        with open(cls.TREE_PATH, 'rb') as t:
            tree = pickle.load(t)
        with open(cls.EDGES_PATH, 'rb') as e:
            edges = pickle.load(e)
        with open(cls.NODES_PATH, 'rb') as n:
            nodes = pickle.load(n)
        return tree, edges, nodes


    @classmethod
    def update_mcts_edges(cls, new_games):
        tree, edges, NODES = cls.get_tree_and_edges()
        for game in new_games:
            win = game[1]
            node = board.Board.arr2str(game[0][0])
            NODES[node]['N'] += 1
            NODES[node][cls.WIN2DICT_MAP[win]] += 1
            NODES[node]['Q'] = NODES[node]['W'] / NODES[node]['N']

            for i in range(len(game[0])-1):
                initial = game[0][i]
                final = game[0][i+1]

                edge = board.Board.arr2str(initial)+'2'+board.Board.arr2str(final)
                edges[edge]['N'] += 1

                node = board.Board.arr2str(final)
                NODES[node]['N'] += 1
                NODES[node][cls.WIN2DICT_MAP[win]] += 1
                NODES[node]['Q'] = NODES[node]['W']/NODES[node]['N']

        cls.save_tree_edges(tree, edges, NODES)

    @classmethod
    def PUCT_function(cls, bool, N, edge):
        if bool == -1:
            Q = 1-edge['Q']
        else:
            Q = edge['Q']

        puct = Q + cls.PUCT_CONSTANT * edge['P'] * np.sqrt(N) / (1+edge['N'])
        return puct


def print_edges(edges):
    for k in edges.keys():
        if edges[k]['N'] == 0:
            print('|{}|'.format(k), edges[k])


def main():
    t, e = MCTS.get_tree_and_edges()
    print_edges(e)

if __name__ == '__main__':
    main()