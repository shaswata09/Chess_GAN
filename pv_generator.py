########################################################
#
# This file can be used by calling
# get_policy(board, generator)
#
# If using the generator provided in the files, then call
# get_policy(board, get_policy_generator(file_path)).
#
# The default file path is "./gen_save", so if using the
# default path, then verify the path exists and use
# get_policy(board, get_policy_generator())
#
# The policy returned will be a list of the form:
#[[move, piece probability, target location probability]]
#
# Example:
# [['g1f3', 0.111, 0.28],
#  ['g1h3', 0.111, 0.27],
#  ['f2f4', 0.110, 0.06],
#  [...]]
#
########################################################


import chess
import data
import math
import numpy as np
import tensorflow as tf

def get_policy_generator(loc = "./gen_save"):
    return tf.keras.models.load_model(loc)

def get_policy(board,  policy_gen):
    # Get the policy from the generator
    pol = policy_gen.predict((data._board_representation(board)).reshape(1,8,8,12))[0]

    # Takes the output of the generator and returns
    # a matrix with the legal moves and a list of legal moves
    pol, moves = _convert_policy(pol, board)

    # Sort the matrices and combine. This will allw for the 
    # consumer of the policy to select and/or pop a move
    pol = _sort_policy(pol, moves)

    return pol

def _convert_policy(policy, board):

    policy = np.multiply(policy,_get_legal_move_mask(board))
    policy = _set_percents(policy)
    # Get the list of legal moves to ensure only legal
    # moves are in the final policy matrix
    moves = list(board.generate_legal_moves())

    # Create an empty matrix to hold the move values
    retval = np.zeros([len(moves), 2])

    idx = 0
    for move in moves:
        mv = str(move)

        # Gets the current position of the piece designated by the move
        # by taking the first two characters of the move (e.g., 'e4')
        # and converting to the numeric position on the board 
        # (e.g., chess.parse_square("e4") =  28
        sqr = chess.parse_square(mv[0:2])

        # Retrieve the value on the current position plane (0)
        retval[idx][0] = policy[sqr//8][sqr%8][0]

        # Retrieve the type of piece on the square in order to 
        # determine which plane to look at for end-state value
        piece = board.piece_at(sqr).symbol()

        # Get the ending location of the pice for the given move
        sqr = chess.parse_square(mv[2:4])

        #### - This section can be optimized if the plane order
        # is changed to utilize the piece type values implemented
        # by python-chess.
        loc_pnt = 0
        if piece == "K":
            retval[idx][1] = policy[sqr//8][sqr % 8][1]
        if piece == "Q":
            retval[idx][1] = policy[sqr//8][sqr % 8][2]
        if piece == "B":
            retval[idx][1] = policy[sqr//8][sqr % 8][3]
        if piece == "N":
            retval[idx][1] = policy[sqr//8][sqr % 8][4]
        if piece == "R":
            retval[idx][1] = policy[sqr//8][sqr % 8][5]
        if piece == "P":
            retval[idx][1] = policy[sqr//8][sqr % 8][6]
        
        idx += 1

    return retval, moves

def _sort_policy(pol, moves):
    retval = []

    # Merge the two list into a single matrix
    for idx in range(len(moves)):
        retval.append([str(moves[idx]), pol[idx][0], pol[idx][1]])

    # Sort the matrix based on piece selection, then target location
    retval.sort(key=lambda row:(-row[1], -row[2]))

    return retval

def _get_legal_move_mask(board):
    
    # Creates an 8x8x7 matrix populated with 0s
    retval = np.zeros((8, 8, 7))

    # Gets a list of legal moves from python-chess
    legal_moves = board.generate_legal_moves()

    # Iterate through each move and populate the matrix

    for move in legal_moves:
        sqr = chess.parse_square(str(move)[:2])

        retval[sqr // 8][sqr % 8][0] = 1

        
        piece = board.piece_at(sqr).symbol()
        sqr = chess.parse_square(str(move)[2:4])
        
        #### - See optimization note above in _convert_policy
        # requires changes to training data generation stage
        if piece == "K":
            retval[sqr // 8][sqr % 8][1] = 1
        if piece == "Q":
            retval[sqr // 8][sqr % 8][2] = 1
        if piece == "B":
            retval[sqr // 8][sqr % 8][3] = 1
        if piece == "N":
            retval[sqr // 8][sqr % 8][4] = 1
        if piece == "R":
            retval[sqr // 8][sqr % 8][5] = 1
        if piece == "P":
            retval[sqr // 8][sqr % 8][6] = 1
    return retval   

def _set_percents(matrix):

    for plane in range(matrix.shape[2]):
        plane_sum = np.sum(matrix[:,:,plane])
        if not plane_sum == 0:
            matrix[:,:,plane] = matrix[:,:,plane]/np.sum(matrix[:,:,plane])
                             
    return matrix 


