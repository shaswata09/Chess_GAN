import numpy as np
import copy
import RLEngine
import UtilFunctions


class TicTacUtils(object):

    @staticmethod
    def get_board_to_nn_input(board):
        return board.flatten()

    @staticmethod
    def get_next_move_board(board_object, player, move):
        new_board = copy.deepcopy(board_object)
        new_board.play_move(player, move)
        return new_board.get_board()

    @staticmethod
    def create_boards_to_nn_input(board_object, player, move):
        vector = np.concatenate((TicTacUtils.get_board_to_nn_input(board_object.get_board()),
                                 TicTacUtils.get_board_to_nn_input(TicTacUtils.get_next_move_board(board_object,
                                                                                                   player, move))))
        return np.ndarray(shape=(18, 1), dtype=float, buffer=vector, order='F')

    @staticmethod
    def get_move_probability(network, board_object, player, move):
        input_vector = TicTacUtils.create_boards_to_nn_input(board_object, player, move)
        out_vector = RLEngine.Network.feedforward(network, input_vector)
        return out_vector[-1], input_vector

    @staticmethod
    def get_moves_probability_vector(network, board_object, player, moves):
        n_moves = len(moves)
        out_prob_vector = np.ndarray(shape=(len(moves), 1), dtype=float, order='F')
        full_inp_vector = []
        for move_index in range(n_moves):
            move_out_prob_vector, move_inp_prob_vector = TicTacUtils.get_move_probability(network, board_object, player,
                                                                                          moves[move_index])
            # print(move_inp_prob_vector)
            out_prob_vector[move_index] = move_out_prob_vector
            full_inp_vector.append(move_inp_prob_vector)

        return out_prob_vector, full_inp_vector

    @staticmethod
    def get_move_to_play(network, board_object):
        move_probability_vector, full_inp_vector = TicTacUtils.get_moves_probability_vector(
            network, board_object, board_object.get_player_to_move(), board_object.get_moves())
        move_index = np.argmax(move_probability_vector)
        return move_index, full_inp_vector

    @staticmethod
    def generate_out_vector(move, n_moves):
        temp = np.ndarray(shape=(2, 1), dtype=float, buffer=np.zeros([2]), order='F')
        temp[0] = 1
        full_out_vector = [temp]*n_moves

        temp1 = copy.deepcopy(temp)
        temp1[0] = 0
        temp1[1] = 1
        full_out_vector[move] = temp1

        return full_out_vector

    @staticmethod
    def generate_inv_out_vector(move, n_moves):
        temp = np.ndarray(shape=(2, 1), dtype=float, buffer=np.zeros([2]), order='F')
        full_out_vector = [temp] * n_moves
        temp1 = copy.deepcopy(temp)
        temp1[0] = 1
        full_out_vector[move] = temp1
        return full_out_vector

    @staticmethod
    def train_network(network, board_object, eta):
        print("Training Network based on Game Data.")
        board_winner = board_object.check_board_winner()
        train_vec = []
        temp = []

        if board_winner == 1:
            win_trainable_vector = board_object.game_record[::2]
            lose_trainable_vector = board_object.game_record[1::2]
        elif board_winner == 0.5:
            win_trainable_vector = board_object.game_record[1::2]
            lose_trainable_vector = board_object.game_record[0::2]
        else:
            win_trainable_vector = board_object.game_record
            lose_trainable_vector = []

        # print(trainable_vector)

        for inp_vectors_list, out_index in win_trainable_vector:
            out_vec = TicTacUtils.generate_out_vector(out_index, len(inp_vectors_list))
            train_vec = train_vec + list(zip(inp_vectors_list, out_vec))

        for inp_vectors_list, out_index in lose_trainable_vector:
            out_vec = TicTacUtils.generate_inv_out_vector(out_index, len(inp_vectors_list))
            train_vec = train_vec + list(zip(inp_vectors_list, out_vec))

        # print(temp)
            # train_vec = train_vec + list(zip(inp_vec, out_vec))
        # print(train_vec)
        network.update_mini_batch(train_vec, eta)
        UtilFunctions.update_network(network)
        return network
