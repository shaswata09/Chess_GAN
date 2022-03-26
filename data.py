import config_env as ce
import chess
import csv
import math
import numpy as np
import random

def _read_csv(file="output_C.csv"):
    ring = 12
    seed = random.randrange(0,12)
    data = {}
    x = 0
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if(len(row)!=0): 
                if(x % ring == seed):
                    data[row[0]] = row[1]
                x += 1
    print(f"Total number of elements added is {x}")
    return data

def _set_percents(matrix):

    for plane in range(matrix.shape[2]):
        plane_sum = np.sum(matrix[:,:,plane])
        if not plane_sum == 0:
            matrix[:,:,plane] = matrix[:,:,plane]/np.sum(matrix[:,:,plane])
                             
    return matrix   

def _get_children(board, data):
    retval = {}

    # Get the list of legal moves from this board
    moves = board.generate_legal_moves()
    # For each move, generate the FEN and place in list
    for move in moves:
        val = 1
        board.push(move)
        fen = board.fen().split(" ")[0]
        if fen in data:
            val = data[fen]
        retval[str(move)] = val
        board.pop()
    # Return the list of FEN
    return retval

def _board_representation(board):
    retval = np.zeros((8, 8, 12))

    for square in chess.SQUARES:
        if board.piece_type_at(square) is None:
            retval[math.floor(square / 8)][square % 8] = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        else: retval[math.floor(square / 8)][square % 8] = ce.chess_dict[board.piece_at(square).symbol()]
    return retval

def _policy_matrix(board, children):

    retval = np.zeros((8, 8, 7))

    for child in children:
        from_pos = chess.parse_square(child[:2])
        to_pos = chess.parse_square(child[2:4])
        piece = board.piece_at(from_pos).symbol()

        retval[math.floor(from_pos /8)][from_pos % 8][0] += float(children[child])

        if piece == "K":
            retval[math.floor(to_pos /8)][to_pos % 8][1] += float(children[child])
        if piece == "Q":
            retval[math.floor(to_pos /8)][to_pos % 8][2] += float(children[child])
        if piece == "B":
            retval[math.floor(to_pos /8)][to_pos % 8][3] += float(children[child])
        if piece == "N":
            retval[math.floor(to_pos /8)][to_pos % 8][4] += float(children[child])
        if piece == "R":
            retval[math.floor(to_pos /8)][to_pos % 8][5] += float(children[child])
        if piece == "P":
            retval[math.floor(to_pos /8)][to_pos % 8][6] += float(children[child])

    return _set_percents(retval)

def _get_boards(data):
    boards = []
    policy = []
    x = 0 
    for row in data:
        #print(f"Processing {x}")
        board = chess.Board(row)
        # Retrieve FEN string for available children
        children = _get_children(board, data)
        # Get the 8 x 8 x 12 representation of the board
        boards.append(_board_representation(board))
        # Generate a policy matrix for the available moves
        policy.append(_policy_matrix(board, children))
        x += 1
    pol = policy[0]

    for i in range(7):
        print(pol[:,:,i])

    return np.array(boards), np.array(policy)

def get_training_data():
    
    return _get_boards(_read_csv())

def get_training_batch(batch_size, boards, policy):
    board_batch = []
    policy_batch = []
    size = boards.shape[0]

    for i in range(batch_size):
        idx = random.randrange(start=0, stop=size)
        board_batch.append(boards[idx])
        policy_batch.append(policy[idx])

    return np.array(board_batch), np.array(policy_batch)
    

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

        retval.append([matrix[math.floor(from_pos /8)][from_pos % 8][0], loc_pnt])
        
    return np.array(retval)

def record_results(file, result):
    f = open(file, "a")
    f.write(result)
    f.write("\n")
    f.close()