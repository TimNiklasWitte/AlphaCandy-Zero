# code based on https://github.com/tobiaskret/iannwtf-project/blob/master/PlayGame.py
# Tim's Flappy Bird project (TensorFlow course WiSe 2021/22)


from Window import *

import sys
sys.path.append("../")
from Config import *
from CandyCrushGym import *

from PolicyValueNetwork import *
from StateToImageConverter import *
from MCTS_Window import *

from WindowUtils import *
from MCTS import *

from threading import Thread

from matplotlib import pyplot as plt

import time 
import argparse
import pyautogui
import imageio

from tkinter.constants import *
import tkinter as tk

import numpy as np
import tensorflow as tf

from PlayGameConfig import *

show_arrow_time = 1
show_swap_time = 1
show_empty_time = 1
drop_candy_time = 0.03


def display_execute_action(action, env, window):      

    #
    # Display arrow
    #

    fieldID = action // env.NUM_DIRECTIONS

    direction = action % env.NUM_DIRECTIONS

    x = fieldID // env.FIELD_SIZE
    y = fieldID % env.FIELD_SIZE



    # top
    if direction == 0:
        img = tk.PhotoImage(file=sys.path[0]+"/Images/Arrows/Top.png")
    # right
    elif direction == 1:
        img = tk.PhotoImage(file=sys.path[0]+"/Images/Arrows/Right.png")
    # down
    elif direction == 2:
        img = tk.PhotoImage(file=sys.path[0]+"/Images/Arrows/Down.png")
    # left
    else:
        img = tk.PhotoImage(file=sys.path[0]+"/Images/Arrows/Left.png")

    # top or down
    if direction == 0 or direction == 2:
        window.display.canvas.create_image(x * window.display.image_size , y * window.display.image_size, image=img, anchor=NW)
                
    # right or left
    else:
        window.display.canvas.create_image(x * window.display.image_size, y * window.display.image_size, image=img, anchor=NW)


    y += env.CANDY_BUFF_HEIGHT

    time.sleep(show_arrow_time)
    img = None

    #
    # Swap
    #

    # Swap candy
    x_swap = x # attention: numpy x->y are swapped
    y_swap = y # attention: numpy x->y are swapped
    # top
    if direction == 0:
        y_swap += -1
    # down
    elif direction == 2: 
        y_swap += 1
    # right 
    elif direction == 1:
        x_swap += 1
    # left 
    elif direction == 3:
        x_swap += -1

    # swap
    tmp = env.state[y,x]
    env.state[y,x] = env.state[y_swap, x_swap]
    env.state[y_swap, x_swap] = tmp

    window.update_game_field()
    time.sleep(show_swap_time)

    #
    # React
    #
    reward = env.react(x,y, x_swap, y_swap)
        
    if reward == 0:
        tmp = env.state[y,x]
        env.state[y,x] = env.state[y_swap, x_swap]
        env.state[y_swap, x_swap] = tmp

        #window.update_policy_plot(reward, action_probs, desired_reward)
        window.update_game_field()

        time.sleep(show_empty_time) # show also undo swap game state

        return 
        
    window.update_game_field()
    #window.update_plots(reward, action_probs, desired_reward)
     
    time.sleep(show_empty_time)

    #
    # Fill 
    #

    columns_to_fill = list(env.columns_to_fill)
    env.columns_to_fill = set()

    while len(columns_to_fill) != 0:
    
        for idx, column_idx in enumerate(columns_to_fill):

            done = True
            for x in range(0, env.FIELD_SIZE + env.CANDY_BUFF_HEIGHT):

                if env.state[x, column_idx] == -1:
                        
                    done = False
                    if x - 1 < 0:
                        candy = np.random.randint(1, env.NUM_NORMAL_CANDIES+1)
                    else:
                        candy = env.state[x - 1, column_idx]
                        env.state[x - 1, column_idx] = -1
        
                    env.state[x, column_idx] = candy
                        
                    time.sleep(drop_candy_time)

                    window.update_game_field()
                    window.display.previous_state[x, column_idx] = candy
                        

            if done:
                columns_to_fill.pop(idx)

    window.update_game_field()

    return reward


def main():

    #
    # Program args handling
    #

    # Set up ArgumentParser
    parser = argparse.ArgumentParser(description="AlphaCandy Zero plays Candy Crush.")
  
    parser.add_argument("--mode", help="Define the window mode (default: \"0\") \"0\" = game window or \"1\" = game window with plots", type=checkMode, required=False, default="0")
    parser.add_argument("--gif", help="File path where the GIF (screenshots of the window) will be saved.", type=is_valid_name, required=False)
    parser.add_argument("--model", help="Set the path to the model weight's which will be loaded (without .index).", type=check_filePath, required=False, default="../saved_model/trained_weights_10")
    parser.add_argument("--steps", help="Set the number of steps which the agent performs.", type=check_step_num, required=False, default="100")
    args = parser.parse_args()

    # Load args
    
    show_plots = False
    if args.mode == "1":
        show_plots = True

    gif_path = ""
    if args.gif != None:
        gif_path = args.gif

    step_num = args.steps
    

    model_weight_path = args.model 

    seed = np.random.randint(0, 500000)
    env =  env = CandyCrushGym(seed=seed,field_size=FIELD_SIZE, num_normal_candies=NUM_NORMAL_CANDIES, candy_buff_height=CANDY_BUFF_HEIGHT)

    reduced_action_space = env.get_reduced_action_space()
    len_reduced_action_space = len(reduced_action_space)
    policyValueNetwork = PolicyValueNetwork(len_reduced_action_space)
    policyValueNetwork.load_weights(model_weight_path).expect_partial()
   

    # isValidAction(4)

    # return 
    stateToImageConverter = StateToImageConverter(FIELD_SIZE, CANDY_BUFF_HEIGHT, CANDY_IMG_SIZE)

    state = env.reset()


    window = Window(env, show_plots)

    window.update_game_field()

    zero_policy = np.zeros(shape=(NUM_ACTIONS,), dtype=np.float32)
    window.update_policy_plot(policy=zero_policy, num_mcts_step=0)
  
    window.update_reward_value_statistics_plot(0, 0, 0, 0, init=True)
    window.update_policy_value_statistics_plot(policy=None, value=None, num_step=None, show=True, init=True)
    
    #
    # Thread: creating gif (periodically create a screenshot)
    #
    thread = Thread(target = record, args = (window, gif_path, ))
    if gif_path != "":
        time.sleep(1)
        thread.start()


    for num_step in range(step_num):
        
        mcts = MCTS_Window(env, policyValueNetwork, stateToImageConverter)

        window.reset_policy_statistics()
   
     
        for num_mcts_step in range(NUM_MCTS_STEPS):
            
            if num_mcts_step % NUM_MCTS_SUB_STEPS_PLOT == 0:
                
                best_action, policy, value = mcts.step(return_policy=True)

                window.update_policy_plot(policy, num_mcts_step)
                window.update_policy_value_statistics_plot(policy, value, num_mcts_step, show=True)

                
            else:
                best_action, policy, value = mcts.step(return_policy=True)
                
                window.update_policy_value_statistics_plot(policy, value, num_mcts_step, show=False)


        best_action, policy, value = mcts.step(return_policy=True)
        
        window.update_policy_value_statistics_plot(policy, value, NUM_MCTS_STEPS, show=True)
        window.update_policy_plot(policy, NUM_MCTS_STEPS)
        
        reward = display_execute_action(best_action, env, window)
 

        state_img = stateToImageConverter(env.state)
        state_img = tf.expand_dims(state_img, axis=0) # add batch dim
        policy, predicated_value = policyValueNetwork(state_img)
        predicated_value = predicated_value[0] # remove batch dim
        window.update_reward_value_statistics_plot(reward, value, predicated_value, num_step)


def record(window, gif_path):
    with imageio.get_writer(gif_path, mode='I') if gif_path != "" else dummy_context_mgr() as gif_writer:

        while True:

            img = get_window_image(window)
           
            gif_writer.append_data(img)
            
            time.sleep(0.5)


def get_window_image(window: Window):
    """
    Create a screenshot of the entire window
    Keyword arguments:
        env -- EnvMananger
    Return:
        screenshot in form of a np.array
    """
    canvas = window.display
    x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
    w, h = canvas.window_width, canvas.window_height  # canvas.winfo_width(), canvas.winfo_height()

    img = pyautogui.screenshot(region=(x, y, w, h))
    img = np.array(img, dtype=np.uint8)

    return np.array(img, dtype=np.uint8)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
