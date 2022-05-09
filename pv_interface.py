#
# This file was setup for the purposes of testing
# It provides access to files with value and policy data the
# networks are trained on.
# 
# Due to the size of the files, it is only recommended this 
# be used during testing and the GANs be stored and accessed
# separately.
#
import chess
import config_env as ce
import data
import numpy as np
import random

fen_dict = None

def get_policy(board, policy_type):
    if policy_type == ce.POLICY_MTX_CARLSEN:
        policy = _get_real_policy(board, ce.FILE_CARLSEN)
    elif policy_type == ce.POLICY_MTX_MULTI:
        policy = _get_real_policy(board, ce.FILE_MULTI)
    return policy



def get_value(board, val_type):

    val = 0
    if val_type == ce.VAL_FROM_FILE:
        val = _get_real_val(board, ce.FILE_VAL)
    if val_type == ce.VAL_RANDOM:
        val = 2*random.random()-1
    return val



# Get a policy matrix from the real data.
def _get_real_policy(board, filename):
    # Read the files
    fen_dict = data._read_csv(filename)
    # Get the moves made
    moves = data._get_made_moves(board, fen_dict)
    # Generate the policy matrix
    policy = data._generate_target_policy_matrix(board, moves)

    # Takes the output of the generator and returns
    # a matrix with the legal moves and a list of legal moves
    pol, moves = _convert_policy(policy, board)

    # Sort the matrices and combine. This will allow for the 
    # consumer of the policy to select and/or pop a move
    pol = _sort_policy(pol, moves)

    return pol

def get_sorted_policy(board, policy):
    # Takes the output of the generator and returns
    # a matrix with the legal moves and a list of legal moves
    pol, moves = _convert_policy(policy, board)

    # Sort the matrices and combine. This will allow for the 
    # consumer of the policy to select and/or pop a move
    pol = _sort_policy(pol, moves)

    return pol

# Converts the policy from an 8x8x7 matrix to a 2 x X matrix
# where X is the total number of legal moves.
def _convert_policy(policy, board):

    policy = np.multiply(policy,_get_legal_move_mask(board))
    policy = data._set_probabilities(policy)
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
        piece = board.piece_type_at(sqr)

        # Get the ending location of the pice for the given move
        sqr = chess.parse_square(mv[2:4])


        retval[idx][1] = policy[sqr//8][sqr % 8][piece]

        
        idx += 1

    return retval, moves

# Creates and 8x8x7 mask with a 1 value for each legal
# move and 0 for all other elements.
def _get_legal_move_mask(board):
    
    # Creates an 8x8x7 matrix populated with 0s
    retval = np.zeros((8, 8, 7))

    # Gets a list of legal moves from python-chess
    legal_moves = board.generate_legal_moves()

    # Iterate through each move and populate the matrix

    for move in legal_moves:
        sqr = chess.parse_square(str(move)[:2])

        retval[sqr // 8][sqr % 8][0] = 1        
        
        piece = board.piece_type_at(sqr)
        
        sqr = chess.parse_square(str(move)[2:4])
        
        retval[sqr //8][sqr % 8][piece] = 1
    return retval   

# Sort the moves based on the product of the move value
# and target value.
def _sort_policy(pol, moves):
    retval = []

    # Merge the two list into a single matrix
    for idx in range(len(moves)):
        retval.append([str(moves[idx]), pol[idx][0] * pol[idx][1]])

    # Sort the matrix based on the product of
    # piece selection and target location
    retval.sort(key=lambda row:(-row[1]))

    x = sum(row[1] for row in retval)

    for row in retval:
        row[1] = row[1]/x

    return retval

# Get a board value from the real data.
def _get_real_val(board, filename):

    # Read the files
    fen_dict = data._read_csv(filename)

    # Gets the FEN representation of the board less
    # information related to castling, en passant
    fen = board.fen().split(" ")[0]

    # If the FEN exists in the data, then the state
    # was reached in the target data
    if fen in fen_dict:
        val = fen_dict[fen]
    # If the FEN doesn't exist, return 0. Given values
    # range between -1 and 1, 0 is neutral.
    else: val = 0

    return val