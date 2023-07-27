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

import numpy as np
import sys

from PlayGameConfig import *

class Display(tk.Frame):

    def __init__(self, master, env, show_plots):

        self.show_plots = show_plots

        # Window contains also plots

        self.image_size = 60

        if show_plots:
            self.window_height = env.FIELD_SIZE * self.image_size + 250
            self.window_width = env.FIELD_SIZE * self.image_size + 1000
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

            self.frame_plots = tk.Frame(master=master)
            self.frame_plots.pack(side=RIGHT, fill=tk.BOTH, expand=True)


            self.fig = Figure(figsize=(6, 6), dpi=100)
            self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.frame_plots)
            self.canvas_plot.get_tk_widget().pack(side=LEFT, fill=tk.BOTH, expand=True)


            self.fig1 = Figure(figsize=(6, 6), dpi=100)
            self.canvas_plot1 = FigureCanvasTkAgg(self.fig1, master=self.frame_plots)
            self.canvas_plot1.get_tk_widget().pack(side=RIGHT, fill=tk.BOTH, expand=True)

        self.step_cnt = 0
        self.collected_rewards = []
        self.steps = []

        self.previous_state = np.zeros_like(env.state)
        self.images = []

        self.candies = []


    def update_game_field(self):
        
        
        for y in range(self.env.FIELD_SIZE):
            for x in range(self.env.FIELD_SIZE):
                candyID = self.env.state[y + self.env.CANDY_BUFF_HEIGHT,x]

                if candyID == -1:
                    self.images[y*self.env.FIELD_SIZE + x] = None
                    self.previous_state[y,x] = -1
                    continue

                if self.previous_state[y,x] == candyID:
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
                
                if self.previous_state[y,x] == 0:
                    self.images.append(image)
                else:
                    self.images[y*self.env.FIELD_SIZE + x] = image

                candy = self.canvas.create_image(x*self.image_size, y*self.image_size, image=image, anchor=NW)
                self.candies.append(candy)

                self.previous_state[y,x] = candyID


  
    def update_policy_plot(self, policy, num_step):

        if not self.show_plots:
            return



        #self.fig.add_subplot(221)
        #policy_probs = np.zeros(shape=(NUM_ACTIONS,), dtype=np.float32)
        actions, probs = zip(*policy)

        
        action_top = np.zeros(shape=(self.env.FIELD_SIZE, self.env.FIELD_SIZE), dtype=np.float32)
        action_right = np.zeros(shape=(self.env.FIELD_SIZE, self.env.FIELD_SIZE), dtype=np.float32)
        action_down = np.zeros(shape=(self.env.FIELD_SIZE, self.env.FIELD_SIZE), dtype=np.float32)
        action_left = np.zeros(shape=(self.env.FIELD_SIZE, self.env.FIELD_SIZE), dtype=np.float32)

        for action_idx, action in enumerate(actions):

            fieldID = action // self.env.NUM_DIRECTIONS

            direction = action % self.env.NUM_DIRECTIONS

            x = fieldID // self.env.FIELD_SIZE
            y = fieldID % self.env.FIELD_SIZE

            # top
            if direction == 0:
                action_top[y,x] = probs[action_idx]
            # down
            elif direction == 2: 
                action_down[y,x] = probs[action_idx]
            # right 
            elif direction == 1:
                action_right[y,x] = probs[action_idx]
            # left 
            elif direction == 3:
                action_left[y,x] = probs[action_idx]


        min_prob = np.min(probs)
        max_prob = np.max(probs)
        
        self.fig.clf()
        self.fig.suptitle(f"Step {num_step}/{NUM_MCTS_STEPS}")

        #
        # Action: Top
        #
        prob_top_plt = self.fig.add_subplot(221)
        prob_top_plt.set_title("Top")
        img = prob_top_plt.imshow(action_top, vmin=min_prob, vmax=max_prob)
    
       
        #
        # Action: Right
        #
        prob_right_plt = self.fig.add_subplot(222)
        prob_right_plt.set_title("Right")
        img = prob_right_plt.imshow(action_right, vmin=min_prob, vmax=max_prob)
        #self.fig.colorbar(img)

        #
        # Desired reward
        #
        # state_value_plt = self.fig.add_subplot(333)
        # state_value_plt.set_title("Desired reward")

        # state_value_plt.bar([1], [desired_reward], align='center')
        # state_value_plt.axes.get_xaxis().set_visible(False)
        # state_value_plt.grid(True)

        #
        # Action: down
        #
        prob_down_plt = self.fig.add_subplot(223)
        prob_down_plt.set_title("Down")
        img = prob_down_plt.imshow(action_down, vmin=min_prob, vmax=max_prob)
        #self.fig.colorbar(img)

        #
        # Action: left
        #
        prob_left_plt = self.fig.add_subplot(224)
        prob_left_plt.set_title("Left")
        img = prob_left_plt.imshow(action_left, vmin=min_prob, vmax=max_prob)
        #self.fig.colorbar(img)

        # self.fig.subplots_adjust(right=0.8)
        # cbar_ax = self.fig.add_axes([1.85, 0.15, 0.05, 0.7])
        # self.fig.colorbar(img, cax=cbar_ax)

        #
        # Received rewards
        #
        # collected_rewards_plt = self.fig.add_subplot(336)

        # if update_stats:
        #     self.steps.append(self.step_cnt)
        #     self.collected_rewards.append(reward)
          

        # collected_rewards_plt.plot(self.steps, self.collected_rewards, label="Reward")
        # collected_rewards_plt.set_xlim(left=max(0, self.step_cnt - 10), right=self.step_cnt + 10)
     
        # collected_rewards_plt.set_title("Obtained rewards")
        # collected_rewards_plt.set_xlabel("Step")
        # collected_rewards_plt.set_ylabel("Reward")
        # collected_rewards_plt.grid(True)
        # collected_rewards_plt.set_ylim(0, 2)

        # if update_stats:
        #     self.step_cnt += 1

        # # Plot mean of collected rewards (not all! only of displayed)
        # collected_rewards_part = self.collected_rewards[max(0, self.step_cnt - 50):self.step_cnt]
        # mean_collected_rewards_part = np.mean(collected_rewards_part)
        # collected_rewards_plt.axhline(mean_collected_rewards_part, color='r', linestyle="--", label="Mean")
        # collected_rewards_plt.legend(loc='lower right')

        # collected_rewards_plt.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.fig.tight_layout()
        self.canvas_plot.draw()