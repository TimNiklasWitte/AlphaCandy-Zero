import tkinter as tk
from tkinter.constants import *

import sys
sys.path.append("../")
from CandyCrushUtiles import *

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from matplotlib.cm import get_cmap

import numpy as np
import sys

from PlayGameConfig import *
from Util import *

class Display(tk.Frame):

    def __init__(self, master, env, show_plots):

        self.show_plots = show_plots

        # Window contains also plots

        self.image_size = 60

        if show_plots:
            self.window_height = env.FIELD_SIZE * self.image_size + 250
            self.window_width = env.FIELD_SIZE * self.image_size + 1250
        else:
            self.window_height = env.FIELD_SIZE * self.image_size
            self.window_width = env.FIELD_SIZE * self.image_size

        self.root_img_path = sys.path[0]+"/Images"
        self.env = env
        

        # Game window
        tk.Frame.__init__(self, master)
        master.geometry(f"{self.window_width}x{self.window_height}")     
        self.canvas = tk.Canvas(width=env.FIELD_SIZE * self.image_size, height=env.FIELD_SIZE * self.image_size, bg='black')
        self.canvas.pack(side=LEFT)

        # Plots
        if show_plots:
            
            
            self.frame_plot = tk.Frame(master=master)
            self.frame_plot.pack(side=RIGHT, fill=tk.BOTH, expand=True)


            # 
            # Policy Plot
            # 
            self.frame_policy_plot = tk.Frame(master=self.frame_plot)
            self.frame_policy_plot.pack(side=LEFT, fill=tk.BOTH, expand=True)
            
            self.fig_policy = Figure(figsize=(6, 6), dpi=100)
            self.canvas_policy_plot = FigureCanvasTkAgg(self.fig_policy, master=self.frame_policy_plot)
            self.canvas_policy_plot.get_tk_widget().pack(side=LEFT, fill=tk.BOTH, expand=True)

            # 
            # Statistics plots
            # 

            ## Top
            self.frame_statistics_plots = tk.Frame(master=self.frame_plot)
            self.frame_statistics_plots.pack(side=RIGHT, fill=tk.BOTH, expand=True)

            # Policy state value statistics
            self.frame_policy_value_statistics_plot = tk.Frame(master=self.frame_statistics_plots)
            self.frame_policy_value_statistics_plot.pack(side=TOP, fill=tk.BOTH, expand=True)

            self.fig_policy_value_statistics = Figure(figsize=(6.5, 4), dpi=100)
            self.canvas_policy_value_statistics_plot = FigureCanvasTkAgg(self.fig_policy_value_statistics, master=self.frame_policy_value_statistics_plot)
            self.canvas_policy_value_statistics_plot.get_tk_widget().pack(side=TOP,fill=tk.BOTH, expand=True)

            ## Down/Bottom
            # Reward value statistics

            self.frame_reward_value_statistics_plot = tk.Frame(master=self.frame_statistics_plots)
            self.frame_reward_value_statistics_plot.pack(side=BOTTOM, fill=tk.BOTH, expand=True)

            self.fig_reward_statistics = Figure(figsize=(4, 4), dpi=100)
            self.canvas_reward_statistics_plot = FigureCanvasTkAgg(self.fig_reward_statistics, master=self.frame_reward_value_statistics_plot)
            self.canvas_reward_statistics_plot.get_tk_widget().pack(side=RIGHT,fill=tk.BOTH, expand=True)

            

        self.step_cnt = 0
        self.collected_rewards = []
        self.steps_reward = []

        self.previous_state = np.zeros_like(env.state)
        self.images = []

        self.candies = []


        self.steps_mcts = []
        self.policy_diffs = []
        self.previous_policy = None

        self.best_action_changed = []
        self.mcts_step_best_action_changed = []
        self.best_action_colors = {}
        self.color_idx = 0

        self.previous_value = 0
        self.value_diffs = []


        self.values = []
        self.predicted_values = []

    def update_game_field(self):
        
        
        for y in range(self.env.FIELD_SIZE):
            for x in range(self.env.FIELD_SIZE):
                candyID = self.env.state[y + self.env.CANDY_BUFF_HEIGHT,x]

                if candyID == -1:
                    self.images[y*self.env.FIELD_SIZE + x] = None
                    self.previous_state[y + self.env.CANDY_BUFF_HEIGHT,x] = -1
                    continue

                if self.previous_state[y + self.env.CANDY_BUFF_HEIGHT,x] == candyID:
                    continue
                
                if isNormalCandy(candyID):
                    file_name = convert_normalCandyID_name(candyID)
                    image = tk.PhotoImage(file=f"{self.root_img_path}/Normal/{file_name}.png")
                
                elif isWrappedCandyID(candyID):
                    candyID = convertWrappedCandy_toNormal(candyID)
                    file_name = convert_normalCandyID_name(candyID)
                    image = tk.PhotoImage(file=f"{self.root_img_path}/Wrapped/{file_name}.png") 
                
                elif isHorizontalStrippedCandy(candyID):
                    candyID = convertHorizontalStrippedCandy_toNormal(candyID)
                    file_name = convert_normalCandyID_name(candyID)
                    image = tk.PhotoImage(file=f"{self.root_img_path}/Striped/Horizontal/{file_name}.png")

                elif isVerticalStrippedCandy(candyID):
                    candyID = convertVerticalStrippedCandy_toNormal(candyID)
                    file_name = convert_normalCandyID_name(candyID)
                    image = tk.PhotoImage(file=f"{self.root_img_path}/Striped/Vertical/{file_name}.png")

                elif candyID == COLOR_BOMB_CANDY_ID:
                    image = tk.PhotoImage(file=f"{self.root_img_path}/ColourBomb/ColourBomb.png")
                
                if self.previous_state[y + self.env.CANDY_BUFF_HEIGHT,x] == 0:
                    self.images.append(image)
                else:
                    self.images[y*self.env.FIELD_SIZE + x] = image

                candy = self.canvas.create_image(x*self.image_size, y*self.image_size, image=image, anchor=NW)
                self.candies.append(candy)

                self.previous_state[y + self.env.CANDY_BUFF_HEIGHT,x] = candyID


    def reset_policy_statistics(self):
        self.steps_mcts = []
        self.previous_policy = None 
        self.policy_diffs = []

        self.previous_value = 0
        self.value_diffs = []

        self.mcts_step_best_action_changed = []

    
    def update_policy_value_statistics_plot(self, policy, value, num_mcts_step, show=False):

    
        #actions, probs = zip(*policy)

        #robs = np.array(probs)
        if len(self.steps_mcts) == 0:
            self.previous_policy = policy
            self.steps_mcts.append(num_mcts_step)

            self.previous_value = value
            return 
        
       
        
      
        best_action_previous_policy = np.argmax(self.previous_policy)
        best_action = np.argmax(policy)

        if best_action != best_action_previous_policy:
            self.mcts_step_best_action_changed.append(num_mcts_step)
            #print(get_x_y_direction(best_action))
                
     
    
        #diff_policy = np.abs(probs - self.previous_policy)
        avg_diff = KL(policy, self.previous_policy)
        self.previous_policy = policy



        #avg_diff = np.sum(diff_policy)

        

        self.policy_diffs.append(avg_diff)
        self.steps_mcts.append(num_mcts_step)

        value_diff = np.abs(value - self.previous_value)

        self.previous_value = value
    
        self.value_diffs.append(value_diff)

        if show:
            self.fig_policy_value_statistics.clf()

            plt_policy_statistics = self.fig_policy_value_statistics.subplots(1)
            plt_policy_statistics.plot(self.steps_mcts[:-1], self.policy_diffs, color="r")

    
            plt_policy_statistics.set_xlim(left=0, right=NUM_MCTS_STEPS)
            plt_policy_statistics.set_ylim(0,1)
            plt_policy_statistics.set_title("Policy changes $\Delta \pi(t) = \mathrm{{KL}}(\pi_t || \pi_{{t-1}})$ \n State value changes $ \Delta v(t) = |v(s)_t - v(s)_{t-1}|$") 
            plt_policy_statistics.set_xlabel("MCTS step t")
            plt_policy_statistics.set_ylabel("$\Delta \pi(t)$", color="r")
            plt_policy_statistics.grid(True)

            plt_value_statistics = plt_policy_statistics.twinx()

            plt_value_statistics.plot(self.steps_mcts[:-1], self.value_diffs, color="b")
            plt_value_statistics.set_ylabel("$\Delta v(t)$", color="b")


            #
            # history of best actions
            #

            if len(self.mcts_step_best_action_changed) != 0:
                mcts_step = self.mcts_step_best_action_changed[0]
                plt_value_statistics.axvline(x=mcts_step, color="darkgrey", label="Best action changed", alpha=0.7)

                for mcts_step in self.mcts_step_best_action_changed[1:]:
                    plt_value_statistics.axvline(x=mcts_step, color="darkgrey", alpha=0.7)

            # idx = 0
            # if len_actions - 5 >= 0:
            #     for mcts_step, best_action in zip(self.mcts_step_best_action_changed[:len_actions - 5], self.best_action_changed_labels[:len_actions - 5]):
            #         plt_value_statistics.axvline(x=mcts_step, color=colors[idx % len_colors])

            #         idx += 1

            # tmp = max(0,len_actions - 5)
            # for mcts_step, best_action in zip(self.mcts_step_best_action_changed[tmp:], self.best_action_changed_labels[tmp:]):
                
            #     plt_value_statistics.axvline(x=mcts_step, color=colors[idx % len_colors], label=best_action)

            #     idx += 1

            plt_value_statistics.legend(title=f"a = ({get_x_y_direction(best_action)})", loc="upper right", prop={'size': 10})



            self.fig_policy_value_statistics.tight_layout()
            self.canvas_policy_value_statistics_plot.draw()



    def update_reward_value_statistics_plot(self, reward, value, predicted_value, num_step, init=False):
        
        self.fig_reward_statistics.clf()

        
        #
        # Reward
        #
        plt_reward_statistics, plt_value_statistics = self.fig_reward_statistics.subplots(1, 2)
        plt_reward_statistics.set_xlim(left=max(0, self.step_cnt - 10), right=self.step_cnt + 10)
     
        plt_reward_statistics.set_title("Obtained rewards")
        plt_reward_statistics.set_xlabel("Step")
        plt_reward_statistics.set_ylabel("Reward")
        plt_reward_statistics.grid(True)

        plt_reward_statistics.xaxis.set_major_locator(MaxNLocator(integer=True))

        #
        # Value
        #
        plt_value_statistics.set_xlim(left=max(0, self.step_cnt - 10), right=self.step_cnt + 10)
     
        plt_value_statistics.set_title("State value")
        plt_value_statistics.set_xlabel("Step")
        plt_value_statistics.set_ylabel("Discounted return")
        plt_value_statistics.grid(True)

        plt_value_statistics.xaxis.set_major_locator(MaxNLocator(integer=True))

        if not init:
            self.collected_rewards.append(reward)
            self.steps_reward.append(num_step)

            self.step_cnt += 1

            #
            # reward
            #

            plt_reward_statistics.plot(self.steps_reward, self.collected_rewards, label="Reward")

    
            # Plot mean of collected rewards (not all! only of displayed)
            collected_rewards_part = self.collected_rewards[max(0, self.step_cnt - 50):self.step_cnt]
            mean_collected_rewards_part = np.mean(collected_rewards_part)
            plt_reward_statistics.axhline(mean_collected_rewards_part, color='r', linestyle="--", label="Mean")
            plt_reward_statistics.legend()

            plt_reward_statistics.xaxis.set_major_locator(MaxNLocator(integer=True))


            self.values.append(value)
            self.predicted_values.append(predicted_value)

            #
            # value
            #
            plt_value_statistics.plot(self.steps_reward, self.values, color="orange", label="Ground truth")
            plt_value_statistics.plot(self.steps_reward, self.predicted_values, color="green", label="Prediction")
            
            plt_value_statistics.legend()

        self.fig_reward_statistics.tight_layout()
        self.canvas_reward_statistics_plot.draw()
  
    def update_policy_plot(self, policy, num_mcts_step):

        if not self.show_plots:
            return

        #actions, probs = zip(*policy)

        
        action_top = np.zeros(shape=(self.env.FIELD_SIZE, self.env.FIELD_SIZE), dtype=np.float32)
        action_right = np.zeros(shape=(self.env.FIELD_SIZE, self.env.FIELD_SIZE), dtype=np.float32)
        action_down = np.zeros(shape=(self.env.FIELD_SIZE, self.env.FIELD_SIZE), dtype=np.float32)
        action_left = np.zeros(shape=(self.env.FIELD_SIZE, self.env.FIELD_SIZE), dtype=np.float32)

        for action in range(NUM_ACTIONS):

            fieldID = action // self.env.NUM_DIRECTIONS

            direction = action % self.env.NUM_DIRECTIONS

            x = fieldID // self.env.FIELD_SIZE
            y = fieldID % self.env.FIELD_SIZE

            # top
            if direction == 0:
                action_top[y,x] = policy[action]
            # down
            elif direction == 2: 
                action_down[y,x] = policy[action]
            # right 
            elif direction == 1:
                action_right[y,x] = policy[action]
            # left 
            elif direction == 3:
                action_left[y,x] = policy[action]


        min_prob = np.min(policy)
        max_prob = np.max(policy)
       
        self.fig_policy.clf()
        self.fig_policy.suptitle(f"Policy $\pi$(a=(x, y, {{top, right, down, left}})|s$)_{{t={num_mcts_step}}}$ \nMCTS step t={num_mcts_step} of {NUM_MCTS_STEPS}")

        #
        # Action: Top
        #
        prob_top_plt = self.fig_policy.add_subplot(221)
        prob_top_plt.set_title("Top")
        prob_top_plt.set_xlabel("x")
        prob_top_plt.set_ylabel("y")
        prob_top_plt.imshow(action_top, vmin=min_prob, vmax=max_prob)
        prob_top_plt.set_xticks(range(FIELD_SIZE))
       
        #
        # Action: Right
        #
        prob_right_plt = self.fig_policy.add_subplot(222)
        prob_right_plt.set_title("Right")
        prob_right_plt.set_xlabel("x")
        prob_right_plt.set_ylabel("y")
        prob_right_plt.imshow(action_right, vmin=min_prob, vmax=max_prob)
        prob_right_plt.set_xticks(range(FIELD_SIZE))

        #
        # Action: down
        #
        prob_down_plt = self.fig_policy.add_subplot(223)
        prob_down_plt.set_title("Down")
        prob_down_plt.set_xlabel("x")
        prob_down_plt.set_ylabel("y")
        prob_down_plt.imshow(action_down, vmin=min_prob, vmax=max_prob)
        prob_down_plt.set_xticks(range(FIELD_SIZE))

        #
        # Action: left
        #
        prob_left_plt = self.fig_policy.add_subplot(224)
        prob_left_plt.set_title("Left")
        prob_left_plt.set_xlabel("x")
        prob_left_plt.set_ylabel("y")
        prob_left_plt.imshow(action_left, vmin=min_prob, vmax=max_prob)
        prob_left_plt.set_xticks(range(FIELD_SIZE))

        self.fig_policy.tight_layout()
        self.canvas_policy_plot.draw()