import TicTacEngine
import UtilFunctions
import TicTacUtils


def play_game_vs_self():
    playing_board = TicTacEngine.TicTacToe()
    playable_moves = playing_board.get_moves()
    player_to_move = playing_board.get_player_to_move()
    network = UtilFunctions.read_network()
    while len(playable_moves) > 0:
        print("Current Board status: ")
        print(playing_board.get_board())
        print("All possible moves: ")
        print(playable_moves)
        move, input_vector = TicTacUtils.TicTacUtils.get_move_to_play(network, playing_board)
        move = int(input(f"Enter the move index for {player_to_move} : "))
        playing_board.play_move(player_to_move, playable_moves[move],  move, input_vector)
        playable_moves = playing_board.get_moves()
        player_to_move = playing_board.get_player_to_move()

    print(f"The game status is: {playing_board.check_board_winner()}")
    TicTacUtils.TicTacUtils.train_network(network, playing_board, 3)


def play_game_vs_comp():
    playing_board = TicTacEngine.TicTacToe()
    player_to_act = int(input("Enter 1 to play as X else press any key."))
    playable_moves = playing_board.get_moves()
    player_to_move = playing_board.get_player_to_move()
    network = UtilFunctions.read_network()

    while len(playable_moves) > 0:
        if player_to_act == 1:
            print("Current Board status: ")
            print(playing_board.get_board())
            print("All possible moves: ")
            print(playable_moves)
            move, input_vector = TicTacUtils.TicTacUtils.get_move_to_play(network, playing_board)
            move = int(input(f"Enter the move index for {player_to_move} : "))
            player_to_act = 0
        else:
            move, input_vector = TicTacUtils.TicTacUtils.get_move_to_play(network, playing_board)
            player_to_act = 1

        playing_board.play_move(player_to_move, playable_moves[move], move, input_vector)
        # print(input_vector)
        playable_moves = playing_board.get_moves()
        player_to_move = playing_board.get_player_to_move()

    print(f"The game status is: {playing_board.check_board_winner()}")
    TicTacUtils.TicTacUtils.train_network(network, playing_board, 3)


def play_game_comp_vs_comp(network):
    playing_board = TicTacEngine.TicTacToe()
    playable_moves = playing_board.get_moves()
    player_to_move = playing_board.get_player_to_move()
    # network = UtilFunctions.read_network()
    move_counter = 0

    while len(playable_moves) > 0:
        # print(f"Board status after move {move_counter}: ")
        # print(playing_board.get_board())
        move, input_vector = TicTacUtils.TicTacUtils.get_move_to_play(network, playing_board)
        playing_board.play_move(player_to_move, playable_moves[move], move, input_vector)
        move_counter = move_counter + 1
        playable_moves = playing_board.get_moves()
        player_to_move = playing_board.get_player_to_move()

    # print(f"Board status after move {move_counter}: ")
    # print(playing_board.get_board())
    # print(f"The game status is: {playing_board.check_board_winner()}")
    network = TicTacUtils.TicTacUtils.train_network(network, playing_board, 0.5)
    return network

