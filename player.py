__author__ = 'Florin Bora'

import os
import numpy as np
import neural_network
import board as b
import mcts
from copy import deepcopy
import keras
import tensorflow as tf

class Random_Player():
    def __init__(self, type, name):
        self.type = type
        self.name = name

    def turn(self, board):
        row, col = np.where(board.board == 0)
        choice = np.random.choice(len(row))
        return self.type, row[choice], col[choice]

    
class Interactive_Player():
    def __init__(self, type, name):
        self.type = type
        self.name = name
        
    def turn(self, board):
        board.display(clear=True)
        while True:
            human_answer = input('Enter move: row, column tuple (indexing starts at 0)')
            move = [int(x) for x in human_answer if str.isdigit(x)]
            row, col = move[0], move[1]
            if len(move) != 2:
                print ('Incorrect input, try again!')
                continue
            if len([x for x in move if x<0 or x>2])>0:
                print('row and column index can have only values: 0, 1, 2')
                continue
            if board.board[row, col] != 0:
                print('Incorrect move; cell must be empty!')
                continue
            break
        return self.type, row, col

    
class Zero_Player():

    def __init__(self, type, name, nn_type, temperature=1):
        self.type = b.Board.STR2INT_MAP[type]
        self.name = name
        self.temperature = temperature
        self.tree, self.edge_statistics, self.nodes = mcts.MCTS.get_tree_and_edges()
        self.value_estimate = ""

    def load_keras(self):
        try:
            self.keras_nn = keras.models.load_model("./best_keras_model.tf")
        except:
            self.keras_nn = neural_network.CNN_Model()

    def nn_turn(self, board):
        possible_moves = np.where(board.board.ravel() == 0)[0]
        Qmax = -100
        best_move = np.random.choice(possible_moves, 1)

        play_map = np.zeros(board.SIZE)

        for move in possible_moves:
            next_board = deepcopy(board)
            next_board.add_move(self.type, *divmod(move, 3))
            x_inp = np.stack([[next_board.board]], -1)
            pred_win = self.keras_nn.predict(x_inp)[0]
            # pred_one_win = pred_win[2]
            # draw = pred_win[1]
            # pred_two_win = pred_win[0]

            pred_one_win = pred_win
            pred_two_win = 1 - pred_win

            if self.type == 1:
                Q = pred_one_win
            else:
                Q = pred_two_win


            if Q > Qmax:
                Qmax = Q
                best_move = move

            row, col = divmod(move, 3)
            play_map[row, col] = Q

        # value_map = play_map.round(2).astype("str")
        # value_map[value_map == "0.0"] = [board.INT2STR_MAP[val] for val in board.board[board.board != 0]]

        # update history
        # self.history["play_map"].append(play_map)
        # self.history["value_map"].append(value_map)

        row, col = divmod(best_move, 3)
        return self.type, row, col

    def turn(self, board):
        if board.empty():
            move_int = np.random.choice(range(9))
        else:
            if self.value_estimate == "nn":
                return self.nn_turn(board)

            illegal_moves = np.where(board.board.ravel()!=0)[0]
            possible_moves = np.where(board.board.ravel()==0)[0]
            x_inp = np.stack([[board.board]], -1)
            pred_winner, prior_prob = self.keras_nn.predict(x_inp)
            prior_prob = prior_prob[0]

            prior_prob[prior_prob < 0] = 1e-6
            np.put(prior_prob, illegal_moves, 0)

            all_possible_states = []
            for move_int in possible_moves:
                row, col = divmod(move_int, 3)
                board_copy = deepcopy(board)
                board_copy.add_move(self.type, row, col)
                next_state = b.Board.arr2str(board_copy.board)
                all_possible_states.append(next_state)

                # state_statistics = self.edge_statistics[next_state]
                # P = prior_prob[idx]
                # state_statistics['P'] = P
                # state_statistics['PUCT'] = mcts.MCTS.PUCT_function(self.type, N, move_statistics[k])

            move_statistics = dict((states, self.nodes[states]) for states in all_possible_states)
            m = list(move_statistics.keys())

            N = sum(v['N'] for v in move_statistics.values())
            for idx in range(len(m)):
                k = m[idx]
                move_int = possible_moves[idx]
                P = prior_prob[move_int]

                move_statistics[k]['P'] = P
                move_statistics[k]['PUCT'] = mcts.MCTS.PUCT_function(self.type, N, move_statistics[k])

            # for player2, want to minimize PUCT
            puct = np.array([move_statistics[k]['PUCT'] for k in m])
            min_puct = np.min(puct)
            if min_puct < 0: puct -= 1.1*min_puct

            if self.temperature > 1e-6:
                temp_adjusted_prob = np.power(puct, 1.0/self.temperature)
                temp_adjusted_prob[temp_adjusted_prob==np.inf] = 1
            else:
                temp_adjusted_prob = np.zeros(len(puct))
                i = np.argmax(puct)
                temp_adjusted_prob[i] = 1

            if sum(temp_adjusted_prob) == 0:
                temp_adjusted_prob[temp_adjusted_prob == 0] = 1

            temp_adjusted_prob /= sum(temp_adjusted_prob)
            move_int = np.random.choice(possible_moves, p=temp_adjusted_prob)
        row, col = divmod(move_int, 3)
        return self.type, row, col


def main():
    player = Random_Player('x', 'Random BOT')
    board = b.Board()
    move = player.turn(board)


if __name__ == '__main__':
    main()
