################################################
# CSE 6633 - AI Project
# Chess AI using GAN and RL
# Team Members:
#    - Keith Strandell
#    - Shaswata Mitra
#    - Ivan Fernandez
#    - David Hertz
#    - Sabin Bhujel
#
################################################

# Project file imports
import config_env as ce

# Standard imports
import numpy as np
import chess
import math

# Convert a chess board into 8x8x13 representation
def chess_board_matrix(board):
    retval = np.zeros((8, 8, 13))

    for square in chess.SQUARES:
        if board.piece_type_at(square) is None:
            retval[math.floor(square / 8)][square % 8] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1])
        else: retval[math.floor(square / 8)][square % 8] = ce.chess_dict[board.piece_at(square).symbol()]
    return retval

# Returns the legal moves in an 8 x 8 x 7 matrix.
def get_legal_move_matrix(board):
    
    # Creates an 8x8x7 matrix populated with 0s
    retval = np.zeros((8, 8, 7))

    # Gets a list of legal moves from python-chess
    legal_moves = board.generate_legal_moves()

    # Iterate through each move and populate the matrix
    if(board.legal_moves.count() == 0):
        DEFAULT_MOVE_VAL = 0
    else: DEFAULT_MOVE_VAL = 100/board.legal_moves.count()
    for move in legal_moves:
        from_pos = chess.parse_square(str(move)[:2])
        to_pos = chess.parse_square(str(move)[2:4])
        piece = board.piece_at(from_pos).symbol()

        retval[math.floor(from_pos /8)][from_pos % 8][0] += DEFAULT_MOVE_VAL

        if piece == "K":
            retval[math.floor(to_pos /8)][to_pos % 8][1] += DEFAULT_MOVE_VAL
        if piece == "Q":
            retval[math.floor(to_pos /8)][to_pos % 8][2] += DEFAULT_MOVE_VAL
        if piece == "B":
            retval[math.floor(to_pos /8)][to_pos % 8][3] += DEFAULT_MOVE_VAL
        if piece == "N":
            retval[math.floor(to_pos /8)][to_pos % 8][4] += DEFAULT_MOVE_VAL
        if piece == "R":
            retval[math.floor(to_pos /8)][to_pos % 8][5] += DEFAULT_MOVE_VAL
        if piece == "P":
            retval[math.floor(to_pos /8)][to_pos % 8][6] += DEFAULT_MOVE_VAL
    return retval

# Generates a mask that ensures only legal moves are in policy matrix. All legal_moves
# moves are tagged as a 1, all others are 0.  Uses same 8 x 8 x 7 matrix format.
def get_legal_move_mask(board):
    
    # Creates an 8x8x7 matrix populated with 0s
    retval = np.zeros((8, 8, 7))

    # Gets a list of legal moves from python-chess
    legal_moves = board.generate_legal_moves()

    # Iterate through each move and populate the matrix

    for move in legal_moves:
        from_pos = chess.parse_square(str(move)[:2])
        to_pos = chess.parse_square(str(move)[2:4])
        piece = board.piece_at(from_pos).symbol()

        retval[math.floor(from_pos /8)][from_pos % 8][0] = 1

        if piece == "K":
            retval[math.floor(to_pos /8)][to_pos % 8][1] = 1
        if piece == "Q":
            retval[math.floor(to_pos /8)][to_pos % 8][2] = 1
        if piece == "B":
            retval[math.floor(to_pos /8)][to_pos % 8][3] = 1
        if piece == "N":
            retval[math.floor(to_pos /8)][to_pos % 8][4] = 1
        if piece == "R":
            retval[math.floor(to_pos /8)][to_pos % 8][5] = 1
        if piece == "P":
            retval[math.floor(to_pos /8)][to_pos % 8][6] = 1
    return retval

# Takes a matrix of moves and returns an array of tuples with move, piece selection probability, destination probability.
def convert_move_matrix(board, matrix):
    legal_moves = board.generate_legal_moves()
    retval = []
    for move in legal_moves:
        from_pos = chess.parse_square(str(move)[:2])
        to_pos = chess.parse_square(str(move)[2:4])
        piece = board.piece_at(from_pos).symbol()
        loc_pnt = 0
        if piece == "K":
            loc_pnt = matrix[math.floor(to_pos /8)][to_pos % 8][1]
        if piece == "Q":
            loc_pnt = matrix[math.floor(to_pos /8)][to_pos % 8][2]
        if piece == "B":
            loc_pnt = matrix[math.floor(to_pos /8)][to_pos % 8][3]
        if piece == "N":
            loc_pnt = matrix[math.floor(to_pos /8)][to_pos % 8][4]
        if piece == "R":
            loc_pnt = matrix[math.floor(to_pos /8)][to_pos % 8][5]
        if piece == "P":
            loc_pnt = matrix[math.floor(to_pos /8)][to_pos % 8][6]

        retval.append([str(move),matrix[math.floor(from_pos /8)][from_pos % 8][0], loc_pnt])
        
    return retval

# Rather than using softmax, builds probabilities with only legal moves (ignores zeros).
def set_percents(matrix):

    for plane in range(matrix.shape[2]):
        plane_sum = np.sum(matrix[:,:,plane])
        if not plane_sum == 0:
            matrix[:,:,plane] = matrix[:,:,plane]/np.sum(matrix[:,:,plane])
                             
    return matrix
    
