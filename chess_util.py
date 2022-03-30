import chess

##### Data Processing and Utility Functions


# Gets a 180 degree rotated representation of the board.  Allows for training from consistent point of view.
def flip_board(board):
    return board.mirror().transform(chess.flip_horizontal)

# Gets the corresponding move from a flipped board.  For example,
# a move from g1 -> f3 would correspond to b8 -> c6
def flip_move(move):
    return chess.Move.from_uci(square_flip(move[:2])+square_flip(move[2:]))


# Takes in a square name and returns the value from 180 degree rotation of the board
#  - e.g., the position a8 on a board that is then rotated 180 degrees will appear to be h1
def square_flip(square):
    file = ord(square[0])
    rank = int(square[1])

    file = file - (2*(4-(104-file))-1)
    rank = rank + (2*(4-rank)+1)

    return  (chr(file) + str(rank))
