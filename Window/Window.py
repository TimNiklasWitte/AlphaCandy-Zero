from threading import Thread, Event
import tkinter as tk

from Display import *


class Window:
    
    def __init__(self, env, show_plots):
        
        self.event = Event()
        

        self.env = env
        self.show_plots = show_plots

        self.windowThread = Thread(target=self.window_loop)
        self.windowThread.start()

        self.event.wait()        

    

    def window_loop(self):

        self.root = tk.Tk()
        
        self.root.title("AlphaCandy Zero")
    
        self.display = Display(master=self.root, env=self.env, show_plots=self.show_plots)

        self.event.set()

        self.display.mainloop()

        
    def update_game_field(self):
        self.display.update_game_field()
     

    def update_policy_plot(self, policy, num_mcts_step):
        self.display.update_policy_plot(policy, num_mcts_step)
    

    def update_policy_statistics_plot(self, policy, num_step, show=False):
        self.display.update_policy_statistics_plot(policy, num_step, show)

    def update_reward_statistics_plot(self, reward, num_step):
        self.display.update_reward_statistics_plot(reward, num_step)

    # def update_mcts_plot(self, reward):
    #     self.