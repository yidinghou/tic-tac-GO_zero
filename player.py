__author__ = 'Florin Bora'

import os
import numpy as np
import neural_network
import board as Board
import mcts
from copy import deepcopy
import keras
from statistics import *

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
        self.type = Board.STR2INT_MAP[type]
        self.name = name
        self.temperature = temperature
        self.tree = ""
        self.value_estimate = ""
        self.keras_nn = ""

    def load_keras(self):
        try:
            self.keras_nn = keras.models.load_model("./best_keras_model.tf")
        except:
            self.keras_nn = neural_network.CNN_Model()

    def nn_turn(self, board):
        possible_moves = np.where(board.board.ravel() == 0)[0]
        Qmax = -100
        best_move = np.random.choice(possible_moves, 1)

        play_map = np.zeros(Board.SIZE)

        for move in possible_moves:
            next_board = deepcopy(board)
            next_board.add_move(self.type, *divmod(move, 3))
            sym_boards = mcts.create_symmetry(next_board.board)
            sym_pred_one_win = []
            sym_pred_two_win = []

            for brd_arr in sym_boards:
                x_inp = np.stack([[brd_arr]], -1)
                pred_win = self.keras_nn.predict(x_inp)[0][0][0]

                # pred_one_win = pred_win[2]
                # draw = pred_win[1]
                # pred_two_win = pred_win[0]

                pred_one_win = pred_win
                # draw = pred_win[1]
                pred_two_win = 1-pred_one_win

                sym_pred_one_win.append(pred_one_win)
                sym_pred_two_win.append(pred_two_win)

            pred_one_win = sum(sym_pred_one_win)
            pred_two_win = sum(sym_pred_two_win)

                # pred_one_win = pred_win
                # pred_two_win = 1 - pred_win

            if self.type == 1:
                Q = - pred_two_win
            else:
                Q = - pred_one_win

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
            pred_winner, prior_prob = 0, np.zeros(9)

            prior_prob[prior_prob < 0] = 1e-6
            np.put(prior_prob, illegal_moves, 0)

            all_possible_state = []
            for move in possible_moves:
                board_copy = deepcopy(board)
                row, col = divmod(move, 3)
                board_copy.add_move(self.type, row, col)
                next_state = Board.arr2str(board_copy.board)
                all_possible_state.append(next_state)

            move_statistics = dict((next_state, mcts.get_node(next_state, self.tree.TREE)) for next_state in all_possible_state)
            m = list(move_statistics.keys())

            for idx in range(len(m)):
                k = m[idx]
                move_int = possible_moves[idx]
                P = prior_prob[move_int]

                node = move_statistics[k]
                # node.P = P
                node.P = node.N
                node.PUCT = mcts.PUCT_function(1, self.type, node)

            # for player2, want to minimize PUCT
            puct = np.array([move_statistics[k].PUCT for k in m])
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
    board = Board.Board()
    move = player.turn(board)


if __name__ == '__main__':
    main()
