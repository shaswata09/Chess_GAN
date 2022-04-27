import TicTacToeGameModes
import UtilFunctions

if __name__ == "__main__":
    # UtilFunctions.create_network([18, 90, 2])
    # # network = RLEngine.Network([18, 90, 2])
    # playing_board = TicTacEngine.TicTacToe()
    # playable_moves = playing_board.get_moves()
    # player_to_move = playing_board.get_player_to_move()
    # print(TicTacUtils.TicTacUtils.create_boards_to_nn_input(playing_board, player_to_move, playable_moves[0]))
    # print(type(TicTacUtils.TicTacUtils.create_boards_to_nn_input(playing_board, player_to_move, playable_moves[0])))
    # print(RLEngine.Network.feedforward(network, TicTacUtils.TicTacUtils.create_boards_to_nn_input(playing_board, player_to_move, playable_moves[0])))
    # print(TicTacUtils.TicTacUtils.get_moves_probability_vector(network, playing_board, player_to_move, playable_moves))
    # print(TicTacUtils.TicTacUtils.get_move_to_play(network, playing_board))

    network = UtilFunctions.read_network()

    # for i in range(1000):
    #     network = TicTacToeGameModes.play_game_comp_vs_comp(network)
    # UtilFunctions.update_network(network)

    TicTacToeGameModes.play_game_vs_comp()
    # TicTacToeGameModes.play_game_vs_self()
