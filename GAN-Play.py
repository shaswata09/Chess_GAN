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
# This file can be run to have the GAN play against
# a random opponent. To traing GAN, the training
# commands need to be uncommented and you will have
# needed to run the preprocessing scripts.
################################################
import pv_interface as pv
import chess
import chess_util
import config_env as ce
import data
import random
import value_network as vn
import tensorflow as tf
import gan



def play_game():
    p_gen = tf.keras.models.load_model(ce.GEN_MODEL_SAVE)
    #v_gen = tf.keras.models.load_model(ce.VAL_MODEL_SAVE)

    for i in range(100):
        board = chess.Board()
        while board.outcome() == None:
            move = get_white_move(p_gen, board)
            board.push(chess.Move.from_uci(move))

            if board.outcome() == None:
                moves = list(board.generate_legal_moves())
                mv = random.choice(moves)
                board.push(mv)
        print(board.outcome())

def get_white_move(gen, board):
    _, i = gen.predict(data._board_representation(board).reshape(1,8,8,12))
    mv_list = pv.get_sorted_policy(board, i[0])
    val = len(mv_list)//10 + 1
    idx = random.randrange(val)
    return(mv_list.pop(idx))[0]

play_game()



#vn.train()
#gan.train()