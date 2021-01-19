__author__='Florin Bora'

import player
import game
import neural_network
import mcts

def train():
    mcts.MCTS.get_tree_and_edges(reset=True)
    # neural_network.nn_predictor.reset_nn_check_pts()
    # nn_training_set = None

    iterations = 2
    player1 = player.Zero_Player('x', 'Bot_ONE', nn_type='best', temperature=1)
    player2 = player.Zero_Player('o', 'Bot_ONE', nn_type='best', temperature=1)

    player1.load_keras()
    player2.load_keras()


    for _ in range(iterations):

        self_play_game = game.Game(player1, player2)
        self_play_results = self_play_game.play(1000)

        augmented_self_play_results = neural_network.augment_data_set(self_play_results)

        mcts.MCTS.update_mcts_edges(augmented_self_play_results)
        tree, edge_statistics = mcts.MCTS.get_tree_and_edges()

        old_model = player2.keras_nn
        new_trained_model = neural_network.train_nn(player1.keras_nn, edge_statistics, iterations=10)

        player1.keras_nn = new_trained_model
        player1.value_estimate = "nn"
        player2.value_estimate = "nn"

        nn_test_game = game.Game(player1, player2)
        wins_player1, wins_player2 = nn_test_game.play_symmetric(100)
        print("P1: %d P2: %d"%(wins_player1, wins_player2))

        if wins_player1 >= wins_player2:
            print("Saving new best model")
            new_trained_model.save("./best_keras_model.tf")
            player2.keras_nn = new_trained_model
            # mcts.MCTS.get_tree_and_edges(reset=True)
        else:
            player1.keras_nn =  old_model



def zero_vs_random():
    N_games = 100
    global_step = 50000
    nn_check_pt = neural_network.nn_predictor.CHECK_POINTS_NAME + '-' + str(global_step)
    
    player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type=nn_check_pt, temperature=0)
    player2 = player.Zero_Player('o', 'Bot_ZERO', nn_type=nn_check_pt, temperature=0)
    player1.value_estimate ="nn"
    player2.value_estimate="nn"

    z_vs_r_game = game.Game(player1, player2)
    w1, w2 = z_vs_r_game.play_symmetric(N_games)
    print('{} vs {} summary:'.format(player1.name, player2.name))
    print('wins={}, draws={}, losses={}'.format(w1, N_games-w1-w2, w2))


def main():
    train()
    # zero_vs_random()

if __name__ == '__main__':
    main()
