########################################################
# Mississippi State University
# CSE 6633: Artificial Intelligence (Spring 2022)
# Group 9
########################################################

import numpy as np
import random
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

import chess
import chess.engine as eng
import chess.svg
#import pandas as pd
import helper_funcs as funcs

#import matplotlib.pyplot as plt
import pickle


def initialize_Q_network(action_data, Q, num_episode = 10):
    action_list, action_map = action_data
    for e in range(num_episode):
        print("Episode, ", e)
        count = 0
        board = chess.Board()
        
        # Make sure White goes first
        #print(board.turn)
        
        while board.outcome() == None:
            count += 1
            #print(f"Start: {count}")
            
            #board.push(chess.Move.from_uci(move))
            if count % 2 == 0:
                # Get training data
                state  = funcs.get_state(board)
                
                tmp_data, valid_idx = funcs.get_valid_moves_data(board, action_map)
                
                invalid_idx = np.setdiff1d(np.arange(funcs.nA), valid_idx)
                
                nSamp  = len(valid_idx) * 3
                nInv   = nSamp - len(valid_idx)
                data_X = np.tile(state, (nSamp, 1))
                data_y = np.zeros((nSamp, funcs.nA))
                
                nValid = len(valid_idx)
                
                data_y[:nValid, :] = tmp_data
                
                out_invalid_idx = np.random.choice(invalid_idx, nInv, replace=False)
                
                for ii, idx in enumerate(out_invalid_idx):
                    data_y[nValid + ii, idx] = funcs.REWARD_INVALID
                
                # Update network
                Q.fit(data_X, data_y, batch_size=256, epochs = 3, shuffle=True, verbose=0)
                
                # Make move
                board.push(random.choice(list(board.generate_legal_moves())))
                
            else:
                board.push(random.choice(list(board.generate_legal_moves())))
            
            out = board.outcome() 
        print(board)
    return Q

def stockfish_Q_network(action_data, Q, num_episode = 10):
    engine = eng.SimpleEngine.popen_uci(r".\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe")
    action_list, action_map = action_data
    for e in range(num_episode):
        print("Episode, ", e)
        count = 0
        board = chess.Board()
        
        # Make sure White goes first
        #print(board.turn)
        
        data_X = []
        data_y = []
        while board.outcome() == None:
            count += 1
            #print(f"Start: {count}")
            
            #board.push(chess.Move.from_uci(move))
            if count % 2 == 0:
                action_arr  = np.zeros(funcs.nA,)

                state  = funcs.get_state(board)
                mv = engine.play(board, chess.engine.Limit(time=0.1))
                mv2 = mv.move
                start = mv2.from_square
                stop  = mv2.to_square
                ostr  = funcs.move_to_string(start, stop)
                idx   = action_map[ostr]
                action_arr[idx] = 5.0
                
                data_X.append(state)
                data_y.append(action_arr)
                
                # Make move
                board.push(mv2)
                
            else:
                board.push(random.choice(list(board.generate_legal_moves())))
        
            
        # Update network
        data_X = np.array(data_X)
        data_y = np.array(data_y)
        Q.fit(data_X, data_y, batch_size=256, epochs=3, shuffle=True, verbose=0)
        print(board)
    return Q
    
def env_reward_function(board, action_list, action):
    query_action = action_list[action]
    is_valid     = funcs.check_valid_move(board, query_action[0], query_action[1])
    
    if is_valid:
        mv = chess.Move(query_action[0], query_action[1])
                
        # Make move
        board.push(mv)
        
        # Check if finished
        if board.outcome() is not None:
            # Finished playing
            done = True
            winner = board.outcome().winner
            if winner is True:
                reward = -10
            elif winner is False:
                reward = 20
            else:
                # draw
                reward = -1
                
        else:
            # Continue playing
            # Make opponent's move
            board.push(random.choice(list(board.generate_legal_moves())))
            
            if board.outcome() is not None:
                # Finished playing
                done = True
                winner = board.outcome().winner
                if winner is True:
                    reward = -10
                elif winner is False:
                    reward = 20
                else:
                    # draw
                    reward = -1
            else:
                done   = False
                reward = -0.03
    else:
        reward = -10
        done   = True
        
        
    nstate  = funcs.get_state(board) 
    return reward, nstate, done
    

def q_learning(action_data, Q, T, gamma=0.99, epsilon=0.999995, epsilon_decay=0.999995):
    engine = eng.SimpleEngine.popen_uci(r".\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe")
    action_list, action_map = action_data
    
    # Number of training samples to store
    memory_len = 50_000
    batch_len = 100
    replay_memory = []
    
    episode_data = []
    
    tau = 20
    
    cnt = 0
    for e in range(2000): #episodes 6000
        board = chess.Board()
        board.push(random.choice(list(board.generate_legal_moves())))
        
        R = 0
        tStart = time.time()
        while True:
            if board.turn != False:
                print("ERROR! White's turn!")
            
            state = funcs.get_state(board)
            cnt += 1
            
            if np.random.random() < epsilon:
                if e % 2 == 0:
                    action = funcs.get_random_move(board, action_map)
                else:
                    try:
                        # Play Generator (Stockfish)
                        mv     = engine.play(board, chess.engine.Limit(time=0.1))
                        mv2    = mv.move
                        start  = mv2.from_square
                        stop   = mv2.to_square
                        ostr   = funcs.move_to_string(start, stop)
                        action = action_map[ostr]
                    except:
                        action = funcs.get_random_move(board, action_map)
                
            else:
                allActions = Q.predict(state.reshape(1, funcs.nS)).flatten()
                action = np.argmax(allActions)
            
            # Environment
            reward, nstate, done = env_reward_function(board, action_list, action)
                
            
            # Add to training data
            replay_memory.append((state, action, reward, nstate, done))
            n = len(replay_memory)
            if n > memory_len:
                replay_memory.pop(0)
            
            # Update
            m = batch_len
            if n <= batch_len:
                m = n
            batch_data = random.sample(replay_memory, m)
            zero_state = np.zeros(funcs.nS)
            
            states_beg = np.array([x[0] for x in batch_data])
            states_end = np.array([(zero_state if x[4] is True else x[3]) for x in batch_data])
            
            # Do a batch predict for speed
            p_beg = T.predict(states_beg) # should be T
            p_end = T.predict(states_end)
    
            x = np.zeros((m, funcs.nS))
            y = np.zeros((m, funcs.nA))
            
            for i in range(m):
                d = batch_data[i]
                s = d[0]
                a = d[1]
                r = d[2]
                #sn = d[3]
                dn = d[4]
                
                t = p_beg[i]
                if dn is True:
                    t[a] = r
                else:
                    t[a] = r + gamma * np.amax(p_end[i])
    
                x[i] = s
                y[i] = t
    
            Q.fit(x, y, batch_size=100, epochs=1, verbose=0)
            
            if cnt % tau == 0:
                T.set_weights(Q.get_weights())
                
            
            #Q[state, action] += alpha * (reward + gamma * Q[nstate].max() * (not done) - Q[state, action])
            
            
            state = nstate
            R += reward
            epsilon *= epsilon_decay
            
            if done:
                break
            
        tEnd = time.time()
        print("Episode: %g, Reward: %f, Epsilon: %f" % (e, R, epsilon))
        episode_data.append([e, R, epsilon, tEnd - tStart])
    return (episode_data, Q)


if __name__ == '__main__':
    run_init_phase      = True
    run_stockfish_phase = False
    run_qlearning_phase = False
    
    #st = reset_game()
    action_data = funcs.get_all_action_data()
    
    if run_init_phase:
        initializer = tf.keras.initializers.Zeros()
        Q = Sequential()
        Q.add(tf.keras.Input(shape=(funcs.nS,)))
        Q.add(Dense(64, activation='relu', kernel_initializer=initializer))
        Q.add(Dense(64, activation='relu', kernel_initializer=initializer))
        Q.add(Dense(funcs.nA, activation='linear', kernel_initializer=initializer))
        #adm = Adam(learning_rate = 0.1)
        adm = Adam()
        Q.compile(loss='mse', optimizer=adm)
        
        dot_img_file = 'model_1.png'
        tf.keras.utils.plot_model(Q, to_file=dot_img_file, show_shapes=True)
        
        # Initialize Random Exploration
        model = initialize_Q_network(action_data, Q, 500)
        model.save("my_h5_model_random_500_7May2022.h5")
    
    if run_stockfish_phase:
        Q = tf.keras.models.load_model("my_h5_model_random_500_7May2022.h5")
        
        model = stockfish_Q_network(action_data, Q, 2000)
        model.save("my_h5_model_random_200_stockfish_2000_7May.h5")
        
        model = stockfish_Q_network(action_data, model, 2000)
        model.save("my_h5_model_random_200_stockfish_4000_7May.h5")
        
        model = stockfish_Q_network(action_data, model, 2000)
        model.save("my_h5_model_random_200_stockfish_6000_7May.h5")
        
    if run_qlearning_phase:
        print('Running Q-learning...')
        Q = tf.keras.models.load_model("my_h5_model_random_200_stockfish_6000_7May_qlearning.h5")
        T = tf.keras.models.load_model("my_h5_model_random_200_stockfish_6000_7May_qlearning.h5")
        out_data, model = q_learning(action_data, Q, T)
        
        model.save("my_h5_model_random_200_stockfish_6000_9May_qlearning_rev2.h5")
        pickle.dump(out_data, open('out_data_rev2_9May_rev2.pkl', 'wb'))
    
    
    print('Done!')
    
# print(Q.summary())