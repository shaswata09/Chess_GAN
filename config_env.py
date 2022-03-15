
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

import numpy as np

##### Data and file paths

PATH_DATA = "./data"
PATH_CHECKPOINT = "./checkpoint"
PATH_MODEL = "./model"
PATH_LOSS = "./loss/" 

D_FILE_LOSS = "d_loss.csv"
G_FILE_LOSS = "g_loss.csv"
FILE_LOSS = "loss.csv" # Not currently used

##### GAN  and Training configuraiton elements
EPOCHS = 100000
BATCH_SIZE = 500
LEARNING_RATE = 0.0001
BETA = 0.9

MAX_NUM_MOVES = 25000
GAME_SELECT_PERCENT = 0.25

# Data Processing elements
WIN_VAL = 1.0
DRAW_VAL = 0.8
LOSE_VAL = 0.6
RAND_VAL = 0.2
DEFAULT_PIECE_SELECT_VAL = .25 #Not currently used
DEFAULT_TARGET_MOVE_VAL = .25 #Not currently used
PGN_MULTIPLIER = 50.0

# Representation of 12 chess piece types as shown in (using numpy instead of python list)
# https://towardsdatascience.com/magnusgan-using-gans-to-play-like-chess-masters-9dded439bc56?gi=254a9d8146a0
# The represtation is consistent with that used in the AlphaZero model
chess_dict = {
    'p' : np.array([1,0,0,0,0,0,0,0,0,0,0,0,0]),
    'P' : np.array([0,0,0,0,0,0,1,0,0,0,0,0,0]),
    'n' : np.array([0,1,0,0,0,0,0,0,0,0,0,0,0]),
    'N' : np.array([0,0,0,0,0,0,0,1,0,0,0,0,0]),
    'b' : np.array([0,0,1,0,0,0,0,0,0,0,0,0,0]),
    'B' : np.array([0,0,0,0,0,0,0,0,1,0,0,0,0]),
    'r' : np.array([0,0,0,1,0,0,0,0,0,0,0,0,0]),
    'R' : np.array([0,0,0,0,0,0,0,0,0,1,0,0,0]),
    'q' : np.array([0,0,0,0,1,0,0,0,0,0,0,0,0]),
    'Q' : np.array([0,0,0,0,0,0,0,0,0,0,1,0,0]),
    'k' : np.array([0,0,0,0,0,1,0,0,0,0,0,0,0]),
    'K' : np.array([0,0,0,0,0,0,0,0,0,0,0,1,0])
    }
