########################################################
# Mississippi State University
# CSE 6633: Artificial Intelligence (Spring 2022)
# Group 9
########################################################


import numpy as np
#import chess_mcts as mcts
import chess


nS = ((8 * 8))
nA = ((8 * 8))**2



#mdl = tf.keras.models.load_model("my_h5_model_random.h5")

##############################################################################
REWARD_INVALID = -10.0
REWARD_VALID   = 1.0

def move_to_string(start, stop):
    str_out = str(start) + ',' + str(stop)
    return str_out

def get_all_action_data():
    actions = []
    actions_map = {}
    cnt = -1
    for i in range(64):
        for j in range(64):
            cnt += 1
            actions.append([i, j])
            
            actions_map[move_to_string(i, j)] = cnt
    return [actions, actions_map]
            

def get_action(board):
    all_act = list(board.generate_legal_moves())
    return all_act

def get_state(board):
    pc_map = board.piece_map()
    
    state = np.zeros(64, dtype=np.int32)
    
    for ix in pc_map:
        pc        = pc_map[ix]
        pc_type   = pc.piece_type
        pc_color  = pc.color
        if pc_color:
            state[ix] = pc_type
        else:
            state[ix] = pc_type + 6
    # state2 = state.reshape(8, 8)   
    state = state / 1.
    return state

def reset_game():
    board = chess.Board()
    state = get_state(board)
    return state
    
def check_valid_move(board, start, stop):
    all_moves = list(board.generate_legal_moves())
    is_valid = False
    for mv in all_moves:
        check1 = mv.from_square == start
        check2 = mv.to_square == stop
        if check1 and check2:
            is_valid = True
            return is_valid
    return is_valid
    
def get_valid_moves_data(board, action_map):
    all_moves  = list(board.generate_legal_moves())
    action_arr = np.zeros((len(all_moves), nA))
    idx_arr    = []
    for i, mv in enumerate(all_moves):
        start = mv.from_square
        stop  = mv.to_square
        ostr  = move_to_string(start, stop)
        idx   = action_map[ostr]
        action_arr[i, idx] = REWARD_VALID
        idx_arr.append(idx)
    idx_arr, ix = np.unique(idx_arr, return_index = True)
    return action_arr[ix, :], idx_arr

def get_random_move(board, action_map):
    all_moves = list(board.generate_legal_moves())
    mv        = np.random.choice(all_moves, 1)[0]
    start     = mv.from_square
    stop      = mv.to_square
    ostr      = move_to_string(start, stop)
    idx       = action_map[ostr]
    return idx

#########################################################################