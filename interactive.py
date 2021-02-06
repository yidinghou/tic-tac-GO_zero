__author__ = 'Florin Bora'
import sys

new_path = [
 '/Users/yidinghou/.local/lib/python3.7/site-packages',
 '/Users/yidinghou/anaconda3/lib/python3.7/site-packages']

sys.path += new_path
import neural_network
import player
import game
import mcts
import keras

def interactive_game():
    mcts.MCTS.PUCT_CONSTANT = 0.0
    # nn_check_pt = neural_network.nn_predictor.CHECK_POINTS_NAME + '-' + str(global_step)
    # player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type=nn_check_pt, temperature=0)
    # player2 = player.Interactive_Player('o', 'Human')
    # player1.keras_nn = keras.models.load_model("./best_keras_model.tf")
    # player1.value_estimate="nn"
    # print(player1.value_estimate)

    player2 = player.Zero_Player('o', 'Bot_ZERO', nn_type="", temperature=0)
    player1 = player.Interactive_Player('x', 'Human')
    player1.value_estimate="nn"
    player2.value_estimate="nn"
    print(player2.value_estimate)
    # player2.keras_nn = keras.models.load_model("./demo_model2.tf")

    player1.keras_nn = keras.models.load_model("./best_keras_model.tf")
    player2.keras_nn = keras.models.load_model("./best_keras_model.tf")

    tree = mcts.MCTS()
    z_v_h_game = game.Game(player1, player2, tree)
    outcome = z_v_h_game.run(simulate=False)
    # print(outcome)
    z_v_h_game.board.display(clear=True)
    if outcome[1] == 0:
        print('Game ended in draw!')
    else:
        winner = player1 if outcome[1] == player1.type else player2
        print('{} won the game!'.format(winner.name))



def main():
    interactive_game()

if __name__ == '__main__':
    main()
