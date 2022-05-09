################################################
# CSE 6633 - AI Project
# Chess AI - GAN files
# Team Members:
#    - Keith Strandell
#    - Shaswata Mitra
#    - Ivan Fernandez
#    - David Hertz
#    - Sabin Bhujel
#
# This file is used for loading preprocessed data
# and building the appropriate training batches
################################################
import config_env as ce

import chess
import csv
import numpy as np
import random



# Reads in the preprocessed data from a file.
# Creates a dictionary based on the csv file.
# The file must be two elements per line:
#  - the first must be a FEN string
#  - the second must be a float
def _read_csv(file=ce.FILE_CARLSEN):
    
    # Initialize the dictionary 
    fen_dict = {}

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Discards empty lines
            if(len(row) != 0):
                # Adds fen and value to dictionary
                fen_dict[row[0]] = row[1]

    return fen_dict


# Gets a training batch from the data in the
# dictionary. 
def generate_training_batch(fen_dict):

    # Instantiates an empty list to hold the board
    # representation associated with each FEN
    # selected for the batch
    boards = []

    # Instantiates an empty list to hold the policy
    # matrix that depicts the move probability for 
    # each FEN selected for the batch
    policies = []

    # Get a subset of the dictionary to process
    batch_dict = dict(random.sample(fen_dict.items(), ce.BATCH_SIZE))

    # Process each FEN in the batch
    for row in batch_dict:
        board = chess.Board(row)
        moves = _get_made_moves(board, fen_dict)
        boards.append(_board_representation(board))
        policies.append(_generate_target_policy_matrix(board, moves))

    return np.array(boards), np.array(policies)

        
#################################################################
#   This section is for functions used to generate the 
#   training batch. 
#################################################################

# Gets the moves made for provided FEN based on preprocessed data
def _get_made_moves(board, fen_dict):
    # Initialize a dictionary to store the moves made from a 
    # given board state
    made_moves = {}

    # Get the list of legal moves from the given board state
    moves = board.generate_legal_moves()

    # Check each move to see if the resulting FEN is in the 
    # preprocessed data
    for move in moves:
        # Default value of 0.001 is being used to represent a 
        # legal move that has not been made according to the
        # preprocessed data.
        val = 0.001
        # Pushes a move onto the stack in order to get the 
        # state of the board after the move
        board.push(move)
        # Gets the FEN representation of the board less
        # information related to castling, en passant
        fen = board.fen().split(" ")[0]
        # If the FEN exists in the data, then the state
        # was reached in the target data
        if fen in fen_dict:
            # Gets the number of times the state was reached
            val = fen_dict[fen]
        # Adds the move to the dictionary along with the number
        # of times the state was reached.
        made_moves[str(move)] = val
        # Pops the last move to return to the original board 
        # state for further processing
        board.pop()

    return made_moves



# Gets a training batch from the data in the
# dictionary. 
def generate_val_training_batch(fen_dict):

    # Instantiates an empty list to hold the board
    # representation associated with each FEN
    # selected for the batch
    boards = []

    # Instantiates an empty list to hold the policy
    # matrix that depicts the move probability for 
    # each FEN selected for the batch
    vals = []

    # Get a subset of the dictionary to process
    batch_dict = dict(random.sample(fen_dict.items(), ce.BATCH_SIZE))

    # Process each FEN in the batch
    for row in batch_dict:
        board = chess.Board(row)
        boards.append(_board_representation(board))
        vals.append(float(fen_dict[row]))

    return np.array(boards), np.array(vals)



# Generates an 8x8x12 representation of a chess board state
def _board_representation(board):

    # Initialize an 8x8x12 matrix with zeros
    retval = np.zeros((8, 8, 12))

    # For each square on the given chess board, get the
    # chess piece and apply the appropriate values from 
    # the dictionary. Each plane in the representation
    # relates to a chess piece type for each side.
    for sqr in chess.SQUARES:
        # If the square returns None, then the sqare is empty and
        # populated with an array of zeros.
        if board.piece_type_at(sqr) is None:
            retval[sqr//8][sqr % 8] = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        # A piece does occupy the square, look to the dictionary to get
        # the appropriate representation
        else: retval[sqr//8][sqr % 8] = ce.chess_dict[board.piece_at(sqr).symbol()]
    return retval

# Generates a policy matrix based on prepocessed data
def _generate_target_policy_matrix(board, moves):

    # Initialize an 8x8x7 matrix of zeros
    retval = np.zeros((8,8,7))

    for move in moves:
        # Get the integer value of the square represented by
        # the first two characters in the move
        sqr = chess.parse_square(move[:2])

        # Assign the value associated with the move to
        # the appropraite location on the 8x8 matrix at
        # plane 0
        retval[sqr//8][sqr%8][0] += float(moves[move])

        # Get the value of the piece type (as defined by python-chess).
        # The value is 1 <= x <= 6.  This will be used to represent
        # the plane for a piece type.
        plane = board.piece_type_at(sqr)

        # Get the square the piece moves to based on the second two
        # characters in the move. It is possible that a move notation
        # has additional characters (e.g., under promotion). These
        # characters are discarded for the purposes of this effort.
        sqr = chess.parse_square(move[2:4])

        # Assign the value associated with the move to the 
        # appropriate location in the piece's plane
        retval[sqr//8][sqr%8][plane] += float(moves[move])

    return _set_probabilities(retval)

# Takes a matrix of preprocessed moves and converts to 
# percentages.
def _set_probabilities(matrix):

    # Get the sum of the moves made by adding
    # the number of moves registered on plane 0
    move_sum = np.sum(matrix[:,:,0])
    # If the sum > 0 then divide all elements by the sum
    # This will yield the percent each element is of the
    # moves made.
    if move_sum > 0:
        for plane in range(matrix.shape[2]):
            matrix[:,:,plane] = matrix[:,:,plane]/move_sum
    return matrix
