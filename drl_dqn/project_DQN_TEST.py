########################################################
# Mississippi State University
# CSE 6633: Artificial Intelligence (Spring 2022)
# Group 9
########################################################


import numpy as np
import random


import tensorflow as tf
import matplotlib.animation as animation

import chess
import chess.svg

import matplotlib.pyplot as plt
import helper_funcs as funcs

from cairosvg import svg2png
from PIL import Image
from io import BytesIO

def animate_DQN(Q):
    action_data = funcs.get_all_action_data()
    action_list, action_map = action_data

    count = 0
    board = chess.Board()
    
    fig, ax = plt.subplots()
    plt.axis('off')
    ims = []
    
    # Make sure White goes first
    #print(board.turn)
    
    while board.outcome() == None:
        count += 1
        #print(f"Start: {count}")
        
        #board.push(chess.Move.from_uci(move))
        if count % 2 == 0:
            state          = funcs.get_state(board)
            _, all_actions = funcs.get_valid_moves_data(board, action_map)
            
            y_pred       = Q.predict(state.reshape(1, -1)).flatten()
            y_pred       = y_pred[all_actions]
            kp           = np.argmax(y_pred)
            action       = all_actions[kp]
            query_action = action_list[action]
            mv = chess.Move(query_action[0], query_action[1])
            
            # Make move
            board.push(mv)
                
        else:
            board.push(random.choice(list(board.generate_legal_moves())))
            
        boardsvg = chess.svg.board(board=board)
        png = svg2png(bytestring=boardsvg)
        pil_img = Image.open(BytesIO(png)).convert('RGBA')
        im = ax.imshow(pil_img, animated=True)
        ims.append([im])
        
    #winner_list.append(board.outcome().winner)
    print(board)
    return [fig,ax, ims, board.outcome().winner]    

def run_animate(Q, name = 'movie_gameplay.mp4'):
    fig, ax, ims, win = animate_DQN(Q)
    print(win)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(name)
    print('Saved a movie!')
    
    
def play_Q_network(action_data, Q, num_episode = 10):
    action_list, action_map = action_data
    winner_list = []
    
    for e in range(num_episode):
        print('Episode, ', e)
        count = 0
        board = chess.Board()
        
        
        # Make sure White goes first
        #print(board.turn)
        
        while board.outcome() == None:
            count += 1
            #print(f"Start: {count}")
            
            #board.push(chess.Move.from_uci(move))
            if count % 2 == 0:
                
                state          = funcs.get_state(board)
                _, all_actions = funcs.get_valid_moves_data(board, action_map)
                
                y_pred       = Q.predict(state.reshape(1, -1)).flatten()
                y_pred       = y_pred[all_actions]
                kp           = np.argmax(y_pred)
                action       = all_actions[kp]
                query_action = action_list[action]
                
                
                mv = chess.Move(query_action[0], query_action[1])
                        
                # Make move
                board.push(mv)
   
            
            else:
                board.push(random.choice(list(board.generate_legal_moves())))
            
        winner_list.append(board.outcome().winner)
        print(board)
        print(board.outcome().winner)
    return winner_list
    

if __name__ == '__main__':
    file_name = 'model_random_100_stockfish_2000.h5'
    Q = tf.keras.models.load_model(file_name)
    action_data = funcs.get_all_action_data()
    #out = play_Q_network(action_data, Q)
    
    run_animate(Q)
    print('Done!')