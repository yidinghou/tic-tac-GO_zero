import numpy as np
import board
import mcts

class Game():

    def __init__(self, player1, player2, tree):
        self.player1 = player1
        self.player2 = player2
        self.current_player = player1
        self.players = [player1, player2]
        self.board = board.Board()
        self.tree = tree

    def reset(self):
        self.current_player = self.player1
        self.board.reset()

    def next(self):
        self.current_player = self.player1 if self.current_player == self.player2 else self.player2

    def run(self, simulate = True):
        self.reset()
        game_states = [self.board.board.copy()]
        # the starting player always has 'x' (i.e. 1)
        self.player1.type = 1
        self.player2.type = -1
        self.player1.tree = self.tree
        self.player2.tree = self.tree
        players = [self.player1, self.player2]
        winner = board.winner(self.board.board)

        game_over = (self.board.full()) or (winner != 0)
        game_states = [self.board.board.copy()]

        while not game_over:
            turns = np.where(self.board.board.ravel() != 0)[0].shape[0]
            current_player = players[turns % 2]
            PATH = {"path": []}
            # new_tree = mcts.MCTS()
            # mcts.simulate(players, turns, self.board, new_tree, PATH, root=self.board, n=100)
            # new_tree.update_nodes()
            # current_player.tree = new_tree
            if simulate:
                mcts.simulate(players, turns, self.board, self.tree, PATH, root=self.board, n=10)
                self.tree.update_nodes()
            move = current_player.turn(self.board)
            self.board.add_move(*move)
            game_states.append(self.board.board.copy())

            winner = board.winner(self.board.board)
            game_over = (self.board.full()) or (winner != 0)

        return [game_states, winner]

    def play(self, N, simulate=True):
        result = []
        for game in range(N):
            result.append(self.run(simulate=simulate))
        # self.tree.save_tree_edges()
        return result

    def play_symmetric(self, N):
        player1_first = np.array([x[1] for x in self.play(N//2, simulate=False)])
        #swap players
        self.player1, self.player2 = self.player2, self.player1
        player1_second = np.array([x[1] for x in self.play(N//2, simulate=False)])
        win1 = (player1_first == 1).sum() + (player1_second == -1).sum()
        win2 = (player1_first == -1).sum() + (player1_second == 1).sum()
        return win1, win2

