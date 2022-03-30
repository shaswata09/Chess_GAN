
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

# Standard imports
import os
import math
import numpy as np
import random

# Project file imports
import shared_func as sf
import config_env as ce
import gan as g

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Python-Chess imports
import chess
import chess.engine



# Selects a random from a list of possilbe moves
def select_move_random(board):
    
    mvs = board.generate_legal_moves()
    
    return random.choice(list(mvs))

# Takes input from the GAN and selects the move with the highest probability
def select_move_gan(board):
    
    i1 = sf.chess_board_matrix(board).reshape(1,8,8,13)
    
    i2 = sf.set_percents(sf.get_legal_move_matrix(board)).reshape(1,8,8,7)
    
    
    matrix = np.array(gen([i1, i2]))
    matrix[0] = np.multiply(matrix[0], sf.get_legal_move_mask(board))
    matrix[0] = sf.set_percents(matrix[0])
    moves = sf.convert_move_matrix(board, matrix[0])

    piece_select_indicies = []
    max_val = moves[0][1]

    for move in moves:
        if move[1] > max_val:
            max_val = move[1]

    idx = 0
    for move in moves:
        if move[1] == max_val:
            piece_select_indicies.append(idx)
        idx += 1

    
    max_val = moves[piece_select_indicies[0]][2]
    idx = piece_select_indicies[0]

    
    for index in piece_select_indicies:
        if move[2] > max_val:
            idx = index
            max_val = move[2]

    return chess.Move.from_uci(moves[idx][0])


# Plays game against stockfish using GAN selection 
def play_stockfish():
    engine = chess.engine.SimpleEngine.popen_uci(r".\sf\stockfish_14.1_win_x64_avx2.exe")
    board = chess.Board()
    player = 0
    while not board.is_game_over() and not board.legal_moves.count() == 0:
        if(board.legal_moves.count() == 0):
            print(board)
        elif player == 0:
            mv = select_move_gan(board)
            board.push(mv)
            print(mv)
            player = 1
        elif player == 1:
            mv = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(mv.move)
            print(mv)
            player = 0
    print(board.outcome())
    return

# Setting up GAN and loading the latest checkpoint.
gen = g.generator()
disc = g.discriminator()
cross_entropy = BinaryCrossentropy(from_logits=True)
gen_op = Adam(learning_rate = ce.LEARNING_RATE, beta_1=ce.BETA)
disc_op = Adam(learning_rate = ce.LEARNING_RATE, beta_1=ce.BETA)
checkpoint_prefix = os.path.join(ce.PATH_CHECKPOINT, "ckpt")
checkpoint = tf.train.Checkpoint(gen_op = gen_op, disc_op = disc_op,
                                         gen = gen, disc = disc)

checkpoint.restore(tf.train.latest_checkpoint(ce.PATH_CHECKPOINT))

# Calls method to play game against stockfish
play_stockfish()

