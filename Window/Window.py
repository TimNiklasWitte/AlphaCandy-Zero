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
    

    def update_policy_value_statistics_plot(self, policy, value, num_step, show=False):
        self.display.update_policy_value_statistics_plot(policy, value, num_step, show)

    def reset_policy_statistics(self):
        self.display.reset_policy_statistics()

    def update_reward_value_statistics_plot(self, reward, value, predicted_value, num_step, init=False):
        self.display.update_reward_value_statistics_plot(reward, value, predicted_value, num_step, init)

