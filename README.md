# AlphaCandy Zero
## Mastering the game of Candy Crush without Human Knowledge

An AI-based on the [AlphaZero algorithm](https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go) is trained to play the game of Candy Crush. The implementation of this AI is based on the [AlphaGo Zero paper](https://www.deepmind.com/publications/mastering-the-game-of-go-without-human-knowledge).

In summary, a modified Monte Carlo Tree Search (MCTS) algorithm uses a neural network for its search. This algorithm returns a policy and a value for the given state.  Subsequently, the neural network learns this policy and this value for this state. Overall, the algorithm and the neural network help each other to become better and better.


<img src="./Media/play_game_mode_0.gif" width="300" height="300">

## Usage

### Directory overview

* **./**: Files for training.
* **Media**: Contains `.png` and `.gif` files belonging to this `README.md`.
* **Plotting scripts**: Scripts creating the plots for the evaluation part of this `README.md`.
    * **Plots**: Plots created by the plotting scripts.
* **Raw MCTS**: Implementation of the vanilla MCTS algorithm, including evaluation scripts. The performance is measured in form of average and sum rewards of an episode of 10 steps. Multiple MCTS algorithms instances (same number of iterations) are executed concurrently.
    * **logs**: 
        * **performance**: The averages of each launched MCTS algorithms instance is stored in a `avg_rewards_X.np` file. 
        In the `sum_rewards_X.np`the sum of rewards of each launched MCTS algorithms instance is stored.
        Note that, `X` indicates the number of iterations of the vanilla MCTS algorithm.
        * **time**: An average duration over 10 steps is determinated how long the algorithm needed for calculating an action. There is an average duration value for each concurrently executed MCTS instance. These averages are stored in a `X.npy` file.
        Same as above, `X` indicates the number of iterations of the vanilla MCTS algorithm.
* **saved_model**: A ``trained_weights_N.index`` file contains the weights of the neural network learning the policy and the value of the MCTS algorithm at each `N`.
* **test_logs**: 
    * **test**: TensorBoard log file - for each epoch, the current train and test loss (including for value and policy) are logged. Besides, the performance of the agent is evaluated - average and sum of rewards. These are stored in this log file.  
* **Window**:
    **Images**: Candy images (e.g. normal, striped etc.) and images of the arrows which highlight the current action: top, down, left and right.
        * **Arrows**, **Normal**, **Striped** etc.


### Training

Run `Training.py` to start the training. The configuration such as number of epochs, number of iterations of the current dataset etc. can be changed in the `Config.py` file.For each epoch, the weights of the neural network will be saved into `./saved_model`. 
Note that, the agent was already trained for 15 epochs. Executing `Training.py` will overwrite the stored weights in `./saved_model`.

```
python3 ./Training.py
```

### Window 

Run `./Window/PlayGame.py` to watch the AlphaCandy Zero play Candy Crush.
Besides that, a GIF can be created, and specific weights can be loaded.

```
python3 ./Window/PlayGame.py
```

```
usage: PlayGame.py [-h] [--mode MODE] [--gif GIF] [--model MODEL]
                   [--steps STEPS]

AlphaCandy Zero plays Candy Crush.

optional arguments:
  -h, --help     show this help message and exit
  --mode MODE    Define the window mode (default: "0") "0" = game window or
                 "1" = game window with plots
  --gif GIF      File path where the GIF (screenshots of the window) will be
                 saved.
  --model MODEL  Set the path to the model weight's which will be loaded
                 (without .index).
  --steps STEPS  Set the number of steps which the agent performs.
```


#### mode = 0

<img src="./Media/play_game_mode_0.gif" width="300" height="300">

#### mode = 1

<img src="./Media/play_game_mode_1.gif" width="1500" height="400">


## Evaluation

### Raw/vanilla MCTS  

#### Average reward

![alt text](./Raw%20MCTS/Plotting%20scripts/Plots/Vanilla_MCTS_performance_avg_reward.png)

#### Sum reward

![alt text](./Raw%20MCTS/Plotting%20scripts/Plots/Vanilla_MCTS_performance_sum_reward.png)

### Train and test loss

![alt text](./Plotting%20scripts/Plots/TrainTestLoss.png)

### Rewards

![alt text](./Plotting%20scripts/Plots/AvgSumReward.png)

#### Distribution of average rewards

![alt text](./Plotting%20scripts/Plots/DistriAvgRewards.png)

#### Distribution of sum rewards

![alt text](./Plotting%20scripts/Plots/DistriSumRewards.png)

## Requirements
- TensorFlow 2
- TensorboardX
- Numpy
- gym
- tkinter
- matplotlib
- argparse
- imageio
- pyautogui
- tqdm
- seaborn
- pandas
- sys 
- PIL
- multiprocessing
- itertools

## Notice

This project is the final project of the *Scientific Programming in Python course* at the University Osnabrueck (summer semester 2023). 
The Candy Crush gym and the window (watch the agent playing the game) were already developed for the final project called [AlphaCandy](https://github.com/TimNiklasWitte/AlphaCandy) of the *Deep Reinforcement Learning* (DRL) course at the University Osnabrueck (summer semester 2022). 
Both were slightly adapted for the MCTS setting. 
Note that, in the final project of the DRL course, three different agents were developed which can play Candy Crush. 
These agents are based on:
* Deep Q-Network
* PPO (Proximal Policy Optimization)
* Decision Transformer