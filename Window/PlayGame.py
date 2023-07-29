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

from Util import *
from Window import *
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
                        candy = np.random.randint(1, env.NUM_ELEMENTS)
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

    # episode_len = 10
    # field_size = 6
    # num_candys = 4
    
    #
    # Program args handling
    #

    # Set up ArgumentParser
    # parser = argparse.ArgumentParser(description="The Decision Transformer plays Candy Crush.")
    # parser.add_argument("--field_size", help="Set the field size (default = 8).", type=checkFieldSize, required=False, default=8)
    # parser.add_argument("--num_candys", help="Set the number of candys (default = 4).", type=checkNumCandys, required=False, default=4)
    # parser.add_argument("--desired_reward", help="Set the desired reward (default = 0.25)", required=False, default=0.25)

    # parser.add_argument("--mode", help="Define the window mode (default: \"0\") \"0\" = game window or \"1\" = game window with plots", type=checkMode, required=False, default="0")
    # parser.add_argument("--gif", help="File path where the GIF (screenshots of the window) will be saved.", required=False)
    

    # args = parser.parse_args()

    # Load args
    # field_size = args.field_size
    # num_candys = args.num_candys
    # desired_reward = args.desired_reward


    # show_plots = False
    # if args.mode == "1":
    #     show_plots = True

    # gif_path = ""
    # if args.gif != None:
    #     gif_path = args.gif

    policyValueNetwork = PolicyValueNetwork()
    policyValueNetwork.load_weights(f"../saved_model/trained_weights_3").expect_partial()
   
    seed = 34326 #np.random.randint(0, 500000)
    print(seed)
    env = CandyCrushGym(seed)

    # isValidAction(4)

    # return 
    stateToImageConverter = StateToImageConverter(FIELD_SIZE, CANDY_BUFF_HEIGHT, CANDY_IMG_SIZE)

    state = env.reset()


    window = Window(env, True)

    window.update_game_field()

    

    # best_action, policy, value, done = mcts.step(return_policy=True)
    # print(list(policy))

    window.update_reward_value_statistics_plot(0, 0, 0, 0, init=True)

    for num_step in range(10):
        
        mcts = MCTS_Window(env, policyValueNetwork, stateToImageConverter)

        window.reset_policy_statistics()
        #window.update_policy_statistics_plot(zip(([], [])), 0, show=True)

        

        for num_mcts_step in range(NUM_MCTS_STEPS):
            
            if num_mcts_step % NUM_MCTS_SUB_STEPS_PLOT == 0:
                
                best_action, policy, value, done = mcts.step(return_policy=True)

                window.update_policy_plot(policy, num_mcts_step)
                window.update_policy_value_statistics_plot(policy, value, num_mcts_step, show=True)

                
            else:
                best_action, policy, value, done = mcts.step(return_policy=True)
                
                window.update_policy_value_statistics_plot(policy, value, num_mcts_step, show=False)


        best_action, policy, value, done = mcts.step(return_policy=True)

        window.update_policy_value_statistics_plot(policy, value, NUM_MCTS_STEPS, show=True)
        window.update_policy_plot(policy, NUM_MCTS_STEPS)
        

     

        print(np.array2string(env.state, separator=","))
        print(best_action)
        reward = display_execute_action(best_action, env, window)
        print("--------")
        print(np.array2string(env.state, separator=","))

        state_img = stateToImageConverter(env.state)
        state_img = tf.expand_dims(state_img, axis=0) # add batch dim
        policy, predicated_value = policyValueNetwork(state_img)
        predicated_value = predicated_value[0] # remove batch dim
        window.update_reward_value_statistics_plot(reward, value, predicated_value, num_step)

    #state = stateToImageConverter(state)

    # plt.imshow(state)
    # plt.show()

    # state_img = stateToImageConverter(state)
    # state_img = tf.expand_dims(state_img, axis=0)
    #policy, value = policyValueNetwork(state_img)

    #policy = policy[0] # remove batch dim
    
 

    #window = Window(env, False)

    #window.update_game_field()
    
    #state, reward, done, _ = env.step(best_action)
    


    

    # print(reward)

    # window.update_game_field()

    # state = env.reset()

    # buff_states = np.zeros(shape=(1, episode_len, field_size, field_size), dtype=np.uint8)
    # buff_actions = np.zeros(shape=(1, episode_len,), dtype=np.uint8)
    # buff_rewards = np.zeros(shape=(1, episode_len,), dtype=np.float32)

    

    # buff_states[0, 0, :, :] = state
    # buff_rewards[0, 0] = desired_reward

    
    # none_action_id = num_actions + 1
    # buff_actions[0, 0:episode_len] = none_action_id

    # window.update_plots(0, np.zeros(shape=(num_actions,)), desired_reward, update_stats=False)

    # #
    # # Thread: creating gif (periodically create a screenshot)
    # #
    # thread = Thread(target = record, args = (window, gif_path, ))
    # if gif_path != "":
    #     thread.start()

    # while True:
            
    #     window.update_game_field()
        
    #     reward = 0
    #     episode_idx = 0
            
    #     cnt_zero = 0

    #     current_state = np.copy(env.state)
    #     while reward == 0:

    #         #
    #         # Preprocess
    #         #

    #         # onehotify states
    #         states = np.reshape(buff_states, newshape=(1*episode_len*field_size*field_size))
    #         num_one_hot = 26 # num of candys
    #         states = tf.one_hot(states, depth=num_one_hot)
    #         states = tf.reshape(states, shape=(1, episode_len, field_size, field_size, num_one_hot))

    #         # onehotify actions
    #         num_actions = field_size*field_size*4 + 1
    #         actions = tf.one_hot(buff_actions, depth=num_actions)

    #         action = decisionTransformer(states, actions, buff_rewards)
    #         action = action[0] # remove batch dim

    #         best_action = np.argmax(action)

    #         # invalid action -> choose valid action
    #         if not env.isValidAction(best_action):
    #             best_action = 2

    #         next_state, reward, _, _ = env.step(best_action)
     
    #         state = next_state

    #         episode_idx += 1

        
    #         if episode_idx < episode_len - 1:

    #             buff_states[0, episode_idx, :, :] = state
    #             buff_rewards[0, episode_idx] = desired_reward
    #             buff_actions[0, episode_idx] = best_action

    #         else:
                    
    #             buff_states[0, 0:episode_len-1, :, :] = buff_states[0, 1:episode_len, :, :]
    #             buff_states[0, -1, :, :] = state

    #             buff_rewards[0, 0:episode_len-1] = buff_rewards[0, 1:episode_len]
    #             buff_rewards[0, -1] = 0.25

    #             buff_actions[0, 0:episode_len-1] = buff_actions[0, 1:episode_len]
    #             buff_actions[0, -1] = best_action


    #         if reward == 0:
    #             cnt_zero += 1
    #         else:
    #             cnt_zero = 0
                
    #         if cnt_zero == episode_len:


    #             buff_states = np.zeros(shape=(1, episode_len, field_size, field_size), dtype=np.uint8)
    #             buff_actions = np.zeros(shape=(1, episode_len,), dtype=np.uint8)
    #             buff_rewards = np.zeros(shape=(1, episode_len,), dtype=np.float32)

    #             buff_states[0, 0, :, :] = state
    #             buff_rewards[0, 0] = desired_reward

    #             none_action_id = field_size*field_size*4 + 1
    #             buff_actions[0, 0:episode_len] = none_action_id

    #             state = env.reset()
    #             current_state = np.copy(env.state)
    #             window.update_game_field()
    #             episode_idx = 0

    #             cnt_zero = 0
      
        

    #     env.state = current_state
    #     display_execute_action(best_action, action[:-1], desired_reward, env, window)



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
