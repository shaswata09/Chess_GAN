import numpy as np
import itertools


class TicTacToe(object):

    def __init__(self):  # initiates the board matrix
        self.board = np.zeros((3, 3), dtype=float)
        self.moves = self.set_moves()
        self.player_to_move = self.set_player_to_move()
        self.game_record = []

    def get_board(self):  # returns current board matrix
        return self.board

    def set_moves(self):  # returns the possible moves in np array
        self.moves = [] if self.check_board_winner() != 0 else np.argwhere(self.board == 0)
        return self.moves

    def get_moves(self):
        return self.moves

    def set_player_to_move(self):  # returns the player to move or 0 if no possible move
        if len(self.get_moves()) == 0 or self.check_board_winner() != 0:
            self.player_to_move = 0
            return 0
        elif len(np.argwhere(self.board == 1)) > len(np.argwhere(self.board == 0.5)):
            self.player_to_move = 0.5
            return 0.5
        else:
            self.player_to_move = 1
            return 1

    def get_player_to_move(self):
        return self.player_to_move

    def play_move(self, player, move, index=None, input_vector=None):
        self.board[move[0], move[1]] = player
        self.game_record.append(tuple([input_vector, index]))
        self.moves = self.set_moves()
        self.player_to_move = self.set_player_to_move()

    @staticmethod
    def check_if_on_same_line(combination):
        if (combination[0][0] == combination[1][0] == combination[2][0] or
                combination[0][1] == combination[1][1] == combination[2][1] or
                ((combination[0][0] == combination[0][1]) and (combination[1][0] == combination[1][1]) and (
                        combination[2][0] == combination[2][1])) or
                (sum(combination[0]) == sum(combination[1]) == sum(combination[2]))):
            return True
        return False

    def check_board_winner(self):
        x_combinations = itertools.combinations(np.argwhere(self.board == 1), 3)
        o_combinations = itertools.combinations(np.argwhere(self.board == 0.5), 3)

        for i in x_combinations:
            if self.check_if_on_same_line(i):
                return 1
        for i in o_combinations:
            if self.check_if_on_same_line(i):
                return 0.5
        return 0

