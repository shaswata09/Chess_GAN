import numpy as np

# Training information
BATCH_SIZE = 32

EPOCHS = 100

VAL_ELEMENTS = 100000
ELEMENTS = 316110

LEARNING_RATE = 0.0001


VAL_MODEL_SAVE = "./val_model"
GEN_MODEL_SAVE = "./gen_model"
DISC_MODEL_SAVE = "./disc_model"

# Representation of 12 chess piece types as shown in (using numpy instead of python list)
# https://towardsdatascience.com/magnusgan-using-gans-to-play-like-chess-masters-9dded439bc56?gi=254a9d8146a0
# The represtation is consistent with that used in the AlphaZero model
chess_dict = {
    'p' : np.array([1,0,0,0,0,0,0,0,0,0,0,0]),
    'P' : np.array([0,0,0,0,0,0,1,0,0,0,0,0]),
    'n' : np.array([0,1,0,0,0,0,0,0,0,0,0,0]),
    'N' : np.array([0,0,0,0,0,0,0,1,0,0,0,0]),
    'b' : np.array([0,0,1,0,0,0,0,0,0,0,0,0]),
    'B' : np.array([0,0,0,0,0,0,0,0,1,0,0,0]),
    'r' : np.array([0,0,0,1,0,0,0,0,0,0,0,0]),
    'R' : np.array([0,0,0,0,0,0,0,0,0,1,0,0]),
    'q' : np.array([0,0,0,0,1,0,0,0,0,0,0,0]),
    'Q' : np.array([0,0,0,0,0,0,0,0,0,0,1,0]),
    'k' : np.array([0,0,0,0,0,1,0,0,0,0,0,0]),
    'K' : np.array([0,0,0,0,0,0,0,0,0,0,0,1])
    }

# Policy matrix source types
POLICY_MTX_CARLSEN = 0
POLICY_MTX_MULTI = 1
POLICY_MTX_GAN = 2

VAL_FROM_FILE = 0
VAL_RANDOM = 1
VAL_FROM_NN = 2

FILE_CARLSEN = "output_C.csv"
FILE_MULTI = "output.csv"
FILE_VAL = "val.csv"

#MCTS
MAX_ROLLOUTS = 1

MAX_DEPTH = 4

EXP_PARAM = 2**.5

WIN = 0
LOSS = 1
DRAW = 2
DEPTH_REACHED = 3