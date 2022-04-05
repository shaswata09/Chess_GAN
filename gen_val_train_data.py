import chess.pgn
import chess
import csv
from pathlib import Path


def flip_board(board):
    return board.mirror().transform(chess.flip_horizontal)

chess_dict = {}
val_dict = {}

path = "./data"

files = list(Path(path).glob("*.pgn"))

win_node = 1


for file in files:
    print(file)
    pgn_file = open(file)
    game = chess.pgn.read_game(pgn_file)

    while game!= None:
        board = chess.Board()
        res = game.headers["Result"]

        # Parses the result such that if white wins, the value is 1
        # if it is a draw, the value is 0 and if black wins, the
        # value is -1
        if not(res == "*"):
            res = str(res.split("-")[0])
            if res == "1/2":
                res = 0
            else: res = 2 * float(res) - 1

        
            game = game.mainline_moves()
            num_moves = len(list(game))
            idx = num_moves
            for move in game:
                board.push(move)
                m_val = res*((num_moves - idx)**0.33)/(num_moves**0.33)
                key = board.fen().split(" ")[0]
                
                if key in val_dict:
                    val_dict[key][0] += m_val
                    val_dict[key][1] += 1
                else: val_dict[key] = [m_val, 1]

                idx -=1    

        game = chess.pgn.read_game(pgn_file)
     
w = csv.writer(open("val_C.csv", "w", newline=''))
for i in val_dict:
    w.writerow([i, val_dict[i][0]/val_dict[i][1]])
